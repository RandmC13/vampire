//------------------------------------------------------------------------------
//
//   This file is part of the VAMPIRE open source package under the
//   GNU GPL (version 2) licence (see licence file for details).
//
//   (c) S Jenkins 2026. All rights reserved.
//
//------------------------------------------------------------------------------
//
//   Active window for the HAMR write (hamr:active-window).
//
//   Without a dipole field, and with grains that share no exchange bonds, each
//   grain evolves only from its own spins, its laser temperature and the head
//   field. Far from the head a grain sits at the minimum temperature with no
//   field, where it cannot switch on the time scale of the write, so it is
//   frozen: its spins are kept and not integrated. A grain is integrated while
//   the laser raises its temperature anywhere by at least
//   hamr:active-window-temperature above the minimum, or it overlaps the head
//   field box, and for a further relaxation time (10 ps) after that, so that it
//   is frozen in equilibrium at the minimum temperature (also when the head
//   jumps to the next track). All grains are integrated for the first 10 ps.
//   The window is not changed while the laser is off.
//
//   Continuous layers (materials with continuous = true, e.g. a cap over the
//   grains) span the film and are exchange coupled to the grains below (their
//   atoms carry the number of the grain they lie over, or of one more grain
//   over the gaps). Their atoms are integrated by position, and the boxes of
//   the grains are those of their other atoms: in columns of 1 nm across
//   the film, the atoms of a column are integrated with their own damping
//   within hamr:active-window-margin of an integrated grain, then over
//   hamr:active-window-buffer with a damping rising smoothly (raised cosine)
//   to hamr:active-window-buffer-damping and the thermal noise scaled by the
//   square root of the damping ratio, so that the buffer stays at its
//   temperature, and are frozen beyond. The buffer absorbs spin waves that
//   would otherwise be reflected by the frozen spins. A grain split by a
//   periodic boundary has its box on whichever side of the boundary gives the
//   smaller box, and is tested at both of its images.
//
//   Window snapshots: while the window runs, only the atoms of grains
//   integrated since the previous snapshot can have changed, so a spin
//   snapshot may hold just those atoms (src/config/atoms_spins.cpp).
//
//------------------------------------------------------------------------------

// C++ standard library headers
#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

// Vampire headers
#include "atoms.hpp"
#include "anisotropy.hpp"
#include "create.hpp"
#include "dipole.hpp"
#include "errors.hpp"
#include "exchange.hpp"
#include "gpu.hpp"
#include "hamr.hpp"
#include "integrate.hpp"
#include "material.hpp"
#include "program.hpp"
#include "sim.hpp"
#include "vio.hpp"
#include "vmpi.hpp"

// hamr headers
#include "internal.hpp"

namespace hamr{

   namespace internal{

      namespace{

      // per grain: bounding box of its magnetic atoms, steps since it was last hot, integrated or not
      std::vector<double> grain_min_x, grain_max_x, grain_min_y, grain_max_y;
      std::vector<uint64_t> grain_cool_steps;
      std::vector<bool> grain_active;
      std::vector<int> active_ranges; // [first, last+1) pairs of integrated atoms
      std::vector<char> grain_changed;  // grain integrated since the last spin snapshot
      bool snapshot_in_window = false;  // a spin snapshot was written while the window was running
      uint64_t window_hold_steps = 0;
      uint64_t num_window_changes = 0;
      bool window_initialised = false;
      std::vector<uint64_t> resume_grain_cool_steps; // steps since each grain was last hot, from a checkpoint

      // marks the window state in a checkpoint file (older files end the HAMR state before it)
      const uint64_t window_state_magic = 0x3157444e49575248ULL;

      // grains without atoms other than of continuous layers, and grains split by a periodic boundary (box
      // beyond the film)
      std::vector<char> grain_is_layer, grain_wraps_x, grain_wraps_y;
      bool has_layers = false;

      // continuous layers: columns of 1 nm across the film, the column of every local layer atom (-1
      // for other atoms), each column's distance to the nearest integrated grain, and whether its atoms
      // were integrated since the last spin snapshot
      const double column_size = 10.0; // Angstroms
      int num_columns_x = 0, num_columns_y = 0;
      std::vector<int> atom_column;
      std::vector<int> local_columns;
      std::vector<double> column_distance;
      std::vector<char> column_changed;

      bool is_magnetic(const int atom){
         return mp::material[atoms::type_array[atom]].non_magnetic == 0;
      }

      bool is_layer_atom(const int atom){
         return atom_column[atom] >= 0;
      }

      // magnetic atom of a continuous layer (before the columns are set up)
      bool in_layer(const int atom){
         return is_magnetic(atom) && mp::material[atoms::type_array[atom]].continuous;
      }

      // distance from p to the interval [lo, hi] (0 inside)
      double interval_distance(const double p, const double lo, const double hi){
         return std::max(0.0, std::max(lo - p, p - hi));
      }

      // the same for a box that may have been moved by one film length L to keep a grain together: the
      // atoms of a split grain lie in the box or in the box moved back by L
      double grain_distance(const double p, const double lo, const double hi, const bool wraps, const double L){
         const double d = interval_distance(p, lo, hi);
         return wraps ? std::min(d, interval_distance(p, lo - L, hi - L)) : d;
      }

      bool grain_overlaps(const double lo, const double hi, const bool wraps, const double L, const double box_lo, const double box_hi){
         return (hi >= box_lo && lo <= box_hi) || (wraps && hi - L >= box_lo && lo - L <= box_hi);
      }

      // distance across a periodic film (nearest image) from p to the interval [lo, hi]
      double periodic_distance(const double p, const double lo, const double hi, const bool periodic, const double L){
         double d = interval_distance(p, lo, hi);
         if(periodic) d = std::min(d, std::min(interval_distance(p, lo - L, hi - L), interval_distance(p, lo + L, hi + L)));
         return d;
      }

      // state of the layer atoms of a column at distance d from the nearest integrated grain
      bool column_integrated(const double d){
         return d <= active_window_margin + active_window_buffer;
      }
      bool column_in_buffer(const double d){
         return d > active_window_margin && d <= active_window_margin + active_window_buffer;
      }

      // distance of every column holding local layer atoms to the nearest integrated grain
      void update_columns(){
         const double Lx = system_dimensions_x, Ly = system_dimensions_y;
         for(const int c : local_columns){
            const double cx = (c % num_columns_x + 0.5)*column_size;
            const double cy = (c / num_columns_x + 0.5)*column_size;
            double d = 1e300;
            for(size_t g = 0; g < grain_active.size(); g++){
               if(!grain_active[g] || grain_is_layer[g]) continue;
               const double dx = periodic_distance(cx, grain_min_x[g], grain_max_x[g], cs::pbc[0], Lx);
               const double dy = periodic_distance(cy, grain_min_y[g], grain_max_y[g], cs::pbc[1], Ly);
               d = std::min(d, std::sqrt(dx*dx + dy*dy));
            }
            column_distance[c] = d;
         }
      }

      // per-atom damping and noise factor of the buffer atoms; the material's values elsewhere
      void update_buffer_damping(){
         if(!window_damping) return;
         const double pi = 3.14159265358979323846;
         for(int atom = 0; atom < atoms::num_atoms; atom++){
            if(!is_layer_atom(atom)) continue;
            const int mat = atoms::type_array[atom];
            const double d = column_distance[atom_column[atom]];
            const bool buffer = column_in_buffer(d);
            window_in_buffer[atom] = buffer;
            window_alpha[atom] = mp::material[mat].alpha;
            if(!buffer){
               window_one_oneplusalpha_sq[atom] = mp::material[mat].one_oneplusalpha_sq;
               window_alpha_oneplusalpha_sq[atom] = mp::material[mat].alpha_oneplusalpha_sq;
               window_noise_scale[atom] = 1.0;
               continue;
            }
            const double alpha_mat = mp::material[mat].alpha;
            const double alpha = alpha_mat + (active_window_buffer_damping - alpha_mat)*0.5*(1.0 - std::cos(pi*(d - active_window_margin)/active_window_buffer));
            window_alpha[atom] = alpha;
            window_one_oneplusalpha_sq[atom] = -mp::material[mat].gamma_rel/(1.0 + alpha*alpha);
            window_alpha_oneplusalpha_sq[atom] = alpha*window_one_oneplusalpha_sq[atom];
            window_noise_scale[atom] = alpha_mat > 0.0 ? std::sqrt(alpha/alpha_mat) : 1.0;
         }
         window_damping_version++;
      }

      bool atom_integrated(const int atom){
         if(!is_magnetic(atom)) return false;
         if(is_layer_atom(atom)) return column_integrated(column_distance[atom_column[atom]]);
         return grain_active[atoms::grain_array[atom]];
      }

      // atoms integrated by this process (halo atoms are copies of other processes' atoms)
      int num_owned_atoms(){
         #ifdef MPICF
            return vmpi::num_core_atoms + vmpi::num_bdry_atoms;
         #else
            return atoms::num_atoms;
         #endif
      }

      void rebuild_active_ranges(){
         num_window_changes++;
         if(has_layers){
            update_columns();
            update_buffer_damping();
         }
         active_ranges.clear();
         bool in_range = false;
         const int num_local = num_owned_atoms();
         for(int atom = 0; atom < num_local; atom++){
            const bool active = atom_integrated(atom);
            if(active && !in_range) active_ranges.push_back(atom);
            if(!active && in_range) active_ranges.push_back(atom);
            in_range = active;
         }
         if(in_range) active_ranges.push_back(num_local);

         int num_grains = 0;
         for(size_t g = 0; g < grain_active.size(); g++) if(grain_active[g] && !grain_is_layer[g]) num_grains++;
         // every process changes the window at the same step, so the total can be summed here
         uint64_t num_atoms = 0;
         for(size_t r = 0; r < active_ranges.size(); r += 2) num_atoms += active_ranges[r+1] - active_ranges[r];
         num_atoms = vmpi::all_reduce_sum(num_atoms);
         zlog << zTs() << "HAMR active window: " << num_grains << " grains, " << num_atoms << " atoms integrated, head at "
              << head_position_x*0.1 << " nm, " << head_position_y*0.1 << " nm" << std::endl;
      }

      } // end of anonymous namespace

      //------------------------------------------------------------------------
      // Refuses simulations the physics argument above does not cover and sets
      // up the grain bounding boxes; called when the HAMR write starts
      //------------------------------------------------------------------------
      void initialize_active_window(){

         if(err::check==true) std::cout << "hamr::internal::initialize_active_window has been called" << std::endl;

         std::vector<std::string> unsupported;
         if(integrate::integrator != integrate::integrator_t::llg_heun) unsupported.push_back("integrators other than llg-heun");
         if(::dipole::activated) unsupported.push_back("dipole fields");
         if(::exchange::get_exchange_type() != 0) unsupported.push_back("vectorial or tensorial exchange");
         if(::exchange::biquadratic || ::exchange::four_spin) unsupported.push_back("biquadratic or four spin exchange");
         if(anisotropy::is_neel_enabled()) unsupported.push_back("Neel anisotropy (couples neighbouring atoms)");
         if(program::lagrange_multiplier_enabled) unsupported.push_back("the Lagrange multiplier");
         if(sim::spin_torque_fields_enabled() || sim::vcma_fields_enabled()) unsupported.push_back("spin torque or voltage controlled anisotropy fields");

         // grains over all processes, and the grains of continuous layers
         const int num_local = num_owned_atoms();
         int num_grains = 0;
         for(int atom = 0; atom < num_local; atom++) if(is_magnetic(atom)) num_grains = std::max(num_grains, atoms::grain_array[atom] + 1);
         #ifdef MPICF
            MPI_Allreduce(MPI_IN_PLACE, &num_grains, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
         #endif
         // grains made only of layer atoms, and whether there are layer atoms at all
         std::vector<int> grain_atoms(num_grains, 0);
         int num_layer_atoms = 0;
         for(int atom = 0; atom < num_local; atom++){
            if(in_layer(atom)) num_layer_atoms++;
            else if(is_magnetic(atom)) grain_atoms[atoms::grain_array[atom]]++;
         }
         #ifdef MPICF
            MPI_Allreduce(MPI_IN_PLACE, grain_atoms.data(), num_grains, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &num_layer_atoms, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
         #endif
         grain_is_layer.assign(num_grains, 0);
         for(int g = 0; g < num_grains; g++) grain_is_layer[g] = grain_atoms[g] == 0;
         has_layers = num_layer_atoms > 0;

         // grains must evolve independently: no exchange between a magnetic atom and an atom outside its grain
         // other than of a continuous layer (same tolerance as the GPU neighbour list); the neighbours of local
         // atoms include halo atoms
         const double tol = gpu::exchange_tolerance;
         int bond_i = -1, bond_j = -1;
         for(int atom = 0; atom < num_local && bond_i < 0; atom++){
            if(!is_magnetic(atom)) continue;
            for(int nn = atoms::neighbour_list_start_index[atom]; nn <= atoms::neighbour_list_end_index[atom]; nn++){
               const int j = atoms::neighbour_list_array[nn];
               if(std::fabs(atoms::i_exchange_list[atoms::neighbour_interaction_type_array[nn]].Jij) <= tol) continue;
               if(!is_magnetic(j)){ bond_i = atom; bond_j = j; break; }
               if(in_layer(atom) || in_layer(j)) continue;
               if(atoms::grain_array[j] != atoms::grain_array[atom]){ bond_i = atom; bond_j = j; break; }
            }
         }
         int bond_found = bond_i >= 0;
         #ifdef MPICF
            MPI_Allreduce(MPI_IN_PLACE, &bond_found, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
         #endif
         if(bond_found){
            std::ostringstream bond;
            bond << "exchange between atoms of different grains (other than of continuous layers) or with non-magnetic atoms";
            if(bond_i >= 0) bond << " (atoms " << bond_i << " and " << bond_j << ")";
            unsupported.push_back(bond.str());
         }

         if(!unsupported.empty()){
            std::string message = "Error: hamr:active-window does not support the following features of this simulation:\n";
            for(size_t i = 0; i < unsupported.size(); i++) message += "\t" + unsupported[i] + "\n";
            message += "Run without hamr:active-window to use them.";
            err::fatal(message);
         }

         // boxes of the grains over all processes, so that every process makes the same decisions: along
         // each periodic direction both the box of the coordinates and the box with the coordinates below
         // half the film moved up by one film length, keeping the smaller (a grain split by the boundary)
         const double L[2] = {system_dimensions_x, system_dimensions_y};
         std::vector<double> box(8*num_grains);
         for(int g = 0; g < num_grains; g++) for(int k = 0; k < 4; k++){ box[8*g + 2*k] = 1e300; box[8*g + 2*k + 1] = 1e300; }
         for(int atom = 0; atom < num_local; atom++){
            if(!is_magnetic(atom) || in_layer(atom)) continue;
            const int g = atoms::grain_array[atom];
            const double c[2] = {atom_coords_x[atom], atom_coords_y[atom]};
            for(int a = 0; a < 2; a++){
               const double moved = c[a] < 0.5*L[a] ? c[a] + L[a] : c[a];
               // minima and negated maxima, so that one MPI_MIN reduces all of them
               box[8*g + 4*a + 0] = std::min(box[8*g + 4*a + 0], c[a]);
               box[8*g + 4*a + 1] = std::min(box[8*g + 4*a + 1], -c[a]);
               box[8*g + 4*a + 2] = std::min(box[8*g + 4*a + 2], moved);
               box[8*g + 4*a + 3] = std::min(box[8*g + 4*a + 3], -moved);
            }
         }
         #ifdef MPICF
            MPI_Allreduce(MPI_IN_PLACE, box.data(), 8*num_grains, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
         #endif
         grain_min_x.assign(num_grains, 0.0); grain_max_x.assign(num_grains, 0.0);
         grain_min_y.assign(num_grains, 0.0); grain_max_y.assign(num_grains, 0.0);
         grain_wraps_x.assign(num_grains, 0); grain_wraps_y.assign(num_grains, 0);
         int num_split = 0;
         for(int g = 0; g < num_grains; g++){
            double* lo[2] = {&grain_min_x[g], &grain_min_y[g]};
            double* hi[2] = {&grain_max_x[g], &grain_max_y[g]};
            char* wraps[2] = {&grain_wraps_x[g], &grain_wraps_y[g]};
            for(int a = 0; a < 2; a++){
               const double lo0 = box[8*g + 4*a + 0], hi0 = -box[8*g + 4*a + 1];
               const double lo1 = box[8*g + 4*a + 2], hi1 = -box[8*g + 4*a + 3];
               // (by more than 1 Angstrom, so that rounding never moves a grain that is not split)
               *wraps[a] = cs::pbc[a] && !grain_is_layer[g] && hi1 - lo1 < hi0 - lo0 - 1.0;
               *lo[a] = *wraps[a] ? lo1 : lo0;
               *hi[a] = *wraps[a] ? hi1 : hi0;
            }
            if(grain_wraps_x[g] || grain_wraps_y[g]) num_split++;
         }

         window_hold_steps = uint64_t(round(1.0e-11/mp::dt_SI));
         grain_cool_steps.assign(num_grains, 0);
         grain_active.assign(num_grains, true);
         grain_changed.assign(num_grains, 1);
         snapshot_in_window = false;
         window_initialised = true;

         // continued simulation: the grains integrated at the checkpoint, as in the simulation that wrote it
         if(!resume_grain_cool_steps.empty()){
            if(resume_grain_cool_steps.size() == grain_cool_steps.size()){
               grain_cool_steps = resume_grain_cool_steps;
               for(int g = 0; g < num_grains; g++) grain_active[g] = grain_cool_steps[g] < window_hold_steps;
               zlog << zTs() << "HAMR active window state loaded from the checkpoint" << std::endl;
            }
            else{
               err::warning("Warning - the checkpoint holds the active window of " + std::to_string(resume_grain_cool_steps.size()) +
                            " grains but the system has " + std::to_string(num_grains) + ", so every grain is integrated for the first 10 ps");
            }
         }

         // continuous layers: the column of every local layer atom, and the per-atom damping of the buffer
         atom_column.assign(atoms::num_atoms, -1);
         if(has_layers){
            num_columns_x = std::max(1, int(std::ceil(system_dimensions_x/column_size)));
            num_columns_y = std::max(1, int(std::ceil(system_dimensions_y/column_size)));
            column_distance.assign(num_columns_x*num_columns_y, 0.0);
            column_changed.assign(num_columns_x*num_columns_y, 1);
            std::vector<char> used(num_columns_x*num_columns_y, 0);
            for(int atom = 0; atom < atoms::num_atoms; atom++){
               if(!in_layer(atom)) continue;
               const int i = std::min(num_columns_x - 1, std::max(0, int(atom_coords_x[atom]/column_size)));
               const int j = std::min(num_columns_y - 1, std::max(0, int(atom_coords_y[atom]/column_size)));
               atom_column[atom] = j*num_columns_x + i;
               used[atom_column[atom]] = 1;
            }
            local_columns.clear();
            for(size_t c = 0; c < used.size(); c++) if(used[c]) local_columns.push_back(c);
            window_damping = active_window_buffer > 0.0;
            if(window_damping){
               window_in_buffer.assign(atoms::num_atoms, 0);
               window_alpha.resize(atoms::num_atoms);
               window_one_oneplusalpha_sq.resize(atoms::num_atoms);
               window_alpha_oneplusalpha_sq.resize(atoms::num_atoms);
               window_noise_scale.assign(atoms::num_atoms, 1.0);
               for(int atom = 0; atom < atoms::num_atoms; atom++){
                  window_alpha[atom] = mp::material[atoms::type_array[atom]].alpha;
                  window_one_oneplusalpha_sq[atom] = mp::material[atoms::type_array[atom]].one_oneplusalpha_sq;
                  window_alpha_oneplusalpha_sq[atom] = mp::material[atoms::type_array[atom]].alpha_oneplusalpha_sq;
               }
            }
         }

         zlog << zTs() << "HAMR active window enabled: " << num_grains << " grains (" << num_split << " split by a periodic boundary), temperature threshold "
              << active_window_dT << " K above the minimum, grains kept for " << window_hold_steps << " steps after leaving it" << std::endl;
         if(has_layers){
            zlog << zTs() << "HAMR active window: continuous layers integrated within " << active_window_margin*0.1 << " nm of integrated grains, then a "
                 << active_window_buffer*0.1 << " nm buffer with the damping rising to " << active_window_buffer_damping << std::endl;
         }
         rebuild_active_ranges();
      }

      //------------------------------------------------------------------------
      // Functions to write and read the window state (steps since each grain was
      // last hot) after the head state in a checkpoint file, so that a continued
      // simulation integrates the same grains as an uninterrupted one
      //------------------------------------------------------------------------
      void save_window_state(std::ostream& chkfile){
         if(!window_initialised) return;
         const uint64_t num_grains = grain_cool_steps.size();
         chkfile.write(reinterpret_cast<const char*>(&window_state_magic), sizeof(uint64_t));
         chkfile.write(reinterpret_cast<const char*>(&num_grains), sizeof(uint64_t));
         chkfile.write(reinterpret_cast<const char*>(grain_cool_steps.data()), sizeof(uint64_t)*num_grains);
      }

      void load_window_state(std::istream& chkfile){
         uint64_t magic = 0;
         uint64_t num_grains = 0;
         chkfile.read(reinterpret_cast<char*>(&magic), sizeof(uint64_t));
         chkfile.read(reinterpret_cast<char*>(&num_grains), sizeof(uint64_t));
         if(!chkfile || magic != window_state_magic) return;
         std::vector<uint64_t> cool_steps(num_grains);
         chkfile.read(reinterpret_cast<char*>(cool_steps.data()), sizeof(uint64_t)*num_grains);
         if(chkfile) resume_grain_cool_steps.swap(cool_steps);
      }

   } // end of internal namespace

   bool active_window_requested(){
      return internal::active_window && program::program == program::program_t::hamr_simulation;
   }

   bool active_window_enabled(){
      return internal::window_initialised;
   }

   //---------------------------------------------------------------------------
   // Spin snapshots with the window: a snapshot may hold only the atoms whose
   // spins changed since the previous snapshot if that snapshot was written
   // while the window was running (every atom moves before the window starts)
   //---------------------------------------------------------------------------
   bool window_snapshot_possible(){
      return internal::window_initialised && internal::snapshot_in_window && !internal::active_window_full_snapshots;
   }

   bool spin_changed_since_snapshot(const int atom){
      if(!internal::is_magnetic(atom)) return false;
      if(internal::is_layer_atom(atom)) return internal::column_changed[internal::atom_column[atom]];
      return internal::grain_changed[atoms::grain_array[atom]];
   }

   void snapshot_written(){
      if(err::check==true) std::cout << "hamr::snapshot_written has been called" << std::endl;
      if(!internal::window_initialised) return;
      internal::snapshot_in_window = true;
      std::fill(internal::grain_changed.begin(), internal::grain_changed.end(), 0);
      std::fill(internal::column_changed.begin(), internal::column_changed.end(), 0);
   }

   uint64_t window_changes(){
      return internal::num_window_changes;
   }

   bool atom_in_window(const int atom){
      return internal::atom_integrated(atom);
   }

   // marks the grains integrated at the current step as changed since the last snapshot
   static void mark_changed_grains(){
      for(size_t g = 0; g < internal::grain_active.size(); g++) if(internal::grain_active[g]) internal::grain_changed[g] = 1;
      for(const int c : internal::local_columns) if(internal::column_integrated(internal::column_distance[c])) internal::column_changed[c] = 1;
   }

   //---------------------------------------------------------------------------
   // Updates the window for the current head position (while the laser is on)
   // and returns the [first, last+1) pairs of the atoms to integrate
   //---------------------------------------------------------------------------
   const std::vector<int>& update_active_window(){

      if(err::check==true) std::cout << "hamr::update_active_window has been called" << std::endl;

      using namespace hamr::internal;
      if(!head_laser_on){
         mark_changed_grains();
         return active_ranges;
      }

      const double px = head_position_x;
      const double py = head_position_y;
      const double one_over_denx = 1.0/(2.0*laser_sigma_x*laser_sigma_x);
      const double one_over_deny = 1.0/(2.0*laser_sigma_y*laser_sigma_y);
      const double DeltaT = Tmax - Tmin;
      // head field box, as in apply_field_spatial_box
      const double box_min_x = px - 0.5*H_bounds_x - NPS;
      const double box_max_x = px + 0.5*H_bounds_x - NPS;
      const double box_min_y = py - 0.5*H_bounds_y;
      const double box_max_y = py + 0.5*H_bounds_y;

      const double Lx = system_dimensions_x, Ly = system_dimensions_y;
      bool changed = false;
      for(size_t g = 0; g < grain_active.size(); g++){
         // continuous layers follow the grains (update_columns)
         if(grain_is_layer[g]) continue;
         // the laser temperature is largest at the point of the bounding box nearest the head
         const double dx = grain_distance(px, grain_min_x[g], grain_max_x[g], grain_wraps_x[g], Lx);
         const double dy = grain_distance(py, grain_min_y[g], grain_max_y[g], grain_wraps_y[g], Ly);
         const bool hot = DeltaT * exp(-dx*dx*one_over_denx) * exp(-dy*dy*one_over_deny) >= active_window_dT;
         const bool in_field = grain_overlaps(grain_min_x[g], grain_max_x[g], grain_wraps_x[g], Lx, box_min_x, box_max_x) &&
                               grain_overlaps(grain_min_y[g], grain_max_y[g], grain_wraps_y[g], Ly, box_min_y, box_max_y);
         if(hot || in_field) grain_cool_steps[g] = 0;
         else if(grain_cool_steps[g] < window_hold_steps) grain_cool_steps[g]++;
         const bool active = grain_cool_steps[g] < window_hold_steps;
         if(active != grain_active[g]){ grain_active[g] = active; changed = true; }
      }
      if(changed) rebuild_active_ranges();
      mark_changed_grains();

      return active_ranges;
   }

} // end of hamr namespace
