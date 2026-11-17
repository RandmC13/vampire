//------------------------------------------------------------------------------
//
//   This file is part of the VAMPIRE open source package under the
//   Free BSD licence (see licence file for details).
//
//   (c) S Jenkins 2026. All rights reserved.
//
//------------------------------------------------------------------------------
//
//   Periodic images for the tensor solver.
//
//   For a system periodic along x and/or y, the field of a cell is the field of
//   the system and of all its copies shifted by multiples of the period. The
//   sum over the copies converges slowly, since a uniformly magnetised film
//   only has no field outside it as a whole, so the moment of each cell is split
//   into the mean moment of its layer (cell grid plane along z) and the rest:
//
//   - the rest is summed over the copies within three periods of each cell
//     (7 x 7 copies for a film periodic in x and y, the nearest copy of every
//     cell in the middle): its field falls off with distance, and for a
//     pattern as long as the period itself the copies left out change the
//     field by less than 0.1 %;
//   - the mean part, a uniform magnetisation of every layer across all grid
//     cells, is the field of an infinite uniformly magnetised film: zero
//     outside the layer, and inside it B = mu0 (Mx, My, 0), mu0 H = (0, 0, -mu0 Mz).
//
//   The tensors of the rest are those of the solver, with the cell pairs within
//   the cutoff computed from the atoms of the nearest copy. The mean part is
//   subtracted from the summed field with the tensors from each cell to every
//   grid cell of each layer in the same copies (at the centre of the moment of
//   the cells with atoms, so that a uniform layer cancels exactly), including
//   the self term of the cell's own grid cell, and replaced by the infinite
//   film.
//
//------------------------------------------------------------------------------

// C++ standard library headers
#include <cmath>
#include <iostream>
#include <sstream>

// Vampire headers
#include "cells.hpp"
#include "constants.hpp"
#include "create.hpp"
#include "dipole.hpp"
#include "errors.hpp"
#include "vio.hpp"

// dipole module headers
#include "internal.hpp"

namespace dipole{

   namespace internal{

      namespace{

         // copies summed cell by cell: within image_rings periods of the nearest copy of each cell
         const int image_rings = 3;
         int rings[2] = {0, 0};

         // index of the copy of a source cell nearest a target cell along axis a (0: x, 1: y), from the
         // difference of their grid indices (target minus source), so that cells half a period apart always
         // choose the same copy whatever the rounding of their positions: the copy c with the difference
         // minus c periods in [-n/2, n/2) for n cells per period
         int nearest_copy(const int target_cell, const int source_cell, const int a){
            if(rings[a] == 0) return 0;
            const int index_t = a == 0 ? target_cell/(grid_n[1]*grid_n[2]) : (target_cell/grid_n[2]) % grid_n[1];
            const int index_s = a == 0 ? source_cell/(grid_n[1]*grid_n[2]) : (source_cell/grid_n[2]) % grid_n[1];
            const int n = period_n[a];
            const int d = 2*(index_t - index_s) + n;    // 2 x (difference + n/2)
            return d >= 0 ? d/(2*n) : -((-d + 2*n - 1)/(2*n));
         }

         // dipole tensor (without prefactor) of a point dipole at displacement r, added to t
         void add_point_tensor(const double rx, const double ry, const double rz, double* t){
            const double rij = 1.0/std::sqrt(rx*rx + ry*ry + rz*rz);
            const double ex = rx*rij;
            const double ey = ry*rij;
            const double ez = rz*rij;
            const double rij3 = rij*rij*rij;
            t[0] += (3.0*ex*ex - 1.0)*rij3;
            t[1] += (3.0*ex*ey      )*rij3;
            t[2] += (3.0*ex*ez      )*rij3;
            t[3] += (3.0*ey*ey - 1.0)*rij3;
            t[4] += (3.0*ey*ez      )*rij3;
            t[5] += (3.0*ez*ez - 1.0)*rij3;
         }

      }

      //------------------------------------------------------------------------
      // Function to decide whether the tensor solver uses periodic images, and
      // to set up the period, the cell grid and the layers holding magnetic atoms
      //------------------------------------------------------------------------
      void initialize_periodic_images(const std::vector<int>& global_atoms_in_cell_count){

         periodic_images = false;
         if(dipole::internal::solver != dipole::internal::tensor) return;
         periodic[0] = cs::pbc[0];
         periodic[1] = cs::pbc[1];
         if(!periodic[0] && !periodic[1]) return;
         periodic_images = true;

         // the cell grid as in cells::initialize; the period in cells
         const double cell_size[3] = {cells::macro_cell_size_x, cells::macro_cell_size_y, cells::macro_cell_size_z};
         for(int a = 0; a < 3; a++) grid_n[a] = int(std::ceil((cs::system_dimensions[a] + 0.01)/cell_size[a]));
         for(int a = 0; a < 2; a++){
            period[a] = cs::system_dimensions[a];
            period_n[a] = std::max(1, int(std::round(period[a]/cell_size[a])));
            if(periodic[a] && std::fabs(period_n[a]*cell_size[a] - period[a]) > 0.01){
               std::ostringstream message;
               message << "Warning - the system size along " << (a == 0 ? "x" : "y") << " (" << period[a] << " A) is not a multiple of the macrocell size ("
                       << cell_size[a] << " A), so the mean moment of each layer of periodic images is spread over " << period_n[a] << " cells";
               err::warning(message.str());
            }
         }

         for(int a = 0; a < 2; a++) rings[a] = periodic[a] ? image_rings : 0;

         // layers (grid planes along z) holding magnetic atoms; cell id = (i*grid_n[1] + j)*grid_n[2] + k
         std::vector<int> layer_has_atoms(grid_n[2], 0);
         for(size_t cell = 0; cell < global_atoms_in_cell_count.size(); cell++){
            if(global_atoms_in_cell_count[cell] > 0) layer_has_atoms[cell % grid_n[2]] = 1;
         }
         magnetic_layers.clear();
         for(int k = 0; k < grid_n[2]; k++) if(layer_has_atoms[k]) magnetic_layers.push_back(k);

         zlog << zTs() << "Dipole tensor solver: periodic images along" << (periodic[0] ? " x" : "") << (periodic[1] ? " y" : "")
              << ", " << (2*rings[0] + 1)*(2*rings[1] + 1) << " copies of each cell summed cell by cell, " << magnetic_layers.size() << " layers with magnetic atoms" << std::endl;

      }

      //------------------------------------------------------------------------
      // Function to add the tensors of the copies of every cell (and of cell
      // celli's own copies) to tensor row row of cell celli
      //------------------------------------------------------------------------
      void add_image_tensors(const int celli,
                             const int row,
                             const double cutoff,
                             const std::vector<int>& global_atoms_in_cell_count,
                             const std::vector<double>& cells_pos_and_mom_array,
                             const std::vector<int>& list_of_cells_with_atoms,
                             const std::vector< std::vector<double> >& atoms_in_cells_array){

         const double xi = cells_pos_and_mom_array[4*celli+0];
         const double yi = cells_pos_and_mom_array[4*celli+1];
         const double zi = cells_pos_and_mom_array[4*celli+2];
         const int num_i_atoms = global_atoms_in_cell_count[celli];

         for(int cellj = 0; cellj < cells_num_cells; cellj++){

            if(global_atoms_in_cell_count[cellj] <= 0) continue;
            const int num_j_atoms = global_atoms_in_cell_count[cellj];
            double t[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

            const double xj = cells_pos_and_mom_array[4*cellj+0];
            const double yj = cells_pos_and_mom_array[4*cellj+1];
            const int cx = nearest_copy(celli, cellj, 0), cy = nearest_copy(celli, cellj, 1);
            for(int ix = cx - rings[0]; ix <= cx + rings[0]; ix++) for(int iy = cy - rings[1]; iy <= cy + rings[1]; iy++){

               if(ix == 0 && iy == 0) continue; // the system itself is in the row already
               const double shift_x = ix*period[0], shift_y = iy*period[1];
               const double rx = xj + shift_x - xi;
               const double ry = yj + shift_y - yi;
               const double rz = cells_pos_and_mom_array[4*cellj+2] - zi;

               // a copy within the cutoff: from the atoms, as compute_inter_tensor
               if(rx*rx + ry*ry + rz*rz <= cutoff*cutoff && num_i_atoms > 0){
                  int ci = -1, cj = -1;
                  for(size_t idx = 0; idx < atoms_in_cells_array.size(); idx++){
                     if(list_of_cells_with_atoms[idx] == celli) ci = idx;
                     if(list_of_cells_with_atoms[idx] == cellj) cj = idx;
                  }
                  if(ci < 0 || cj < 0){
                     std::cerr << "Programmer error! cell " << (ci < 0 ? celli : cellj) << " is not found in list of local cells with atomic positions!" << std::endl;
                     continue;
                  }
                  double ta[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                  for(int pi = 0; pi < num_i_atoms; pi++){
                     const double cix = atoms_in_cells_array[ci][4*pi+0];
                     const double ciy = atoms_in_cells_array[ci][4*pi+1];
                     const double ciz = atoms_in_cells_array[ci][4*pi+2];
                     for(int qj = 0; qj < num_j_atoms; qj++){
                        add_point_tensor(atoms_in_cells_array[cj][4*qj+0] + shift_x - cix,
                                         atoms_in_cells_array[cj][4*qj+1] + shift_y - ciy,
                                         atoms_in_cells_array[cj][4*qj+2] - ciz, ta);
                     }
                  }
                  const double inorm = 1.0/(double(num_i_atoms)*double(num_j_atoms));
                  for(int c = 0; c < 6; c++) t[c] += ta[c]*inorm;
               }
               // otherwise point dipoles at the cell centres
               else add_point_tensor(rx, ry, rz, t);

            }

            rij_tensor_xx[row][cellj] += t[0];
            rij_tensor_xy[row][cellj] += t[1];
            rij_tensor_xz[row][cellj] += t[2];
            rij_tensor_yy[row][cellj] += t[3];
            rij_tensor_yz[row][cellj] += t[4];
            rij_tensor_zz[row][cellj] += t[5];

         }

      }

      //------------------------------------------------------------------------
      // Function to calculate, for every local cell in which the field is
      // calculated, the tensors from the cell to the centres of all grid cells of
      // each magnetic layer in the system and its nearest copies (without the
      // cell's own grid cell, whose self term is added in add_uniform_field)
      //------------------------------------------------------------------------
      void initialize_uniform_tensors(const std::vector<int>& global_atoms_in_cell_count,
                                      const std::vector<double>& cells_pos_and_mom_array){

         if(!periodic_images) return;
         const int nl = magnetic_layers.size();
         uniform_tensor.assign(6*nl*cells_num_local_cells, 0.0);
         const double cell_size[3] = {cells::macro_cell_size_x, cells::macro_cell_size_y, cells::macro_cell_size_z};

         // position of the mean moment of every grid cell of the magnetic layers: the centre of the moment of a
         // cell with magnetic atoms (where the solver puts its moment, so that a uniform layer cancels exactly),
         // and for an empty cell its centre moved by the mean offset of the cells with atoms in its layer
         std::vector<double> offset(3*nl, 0.0);
         std::vector<int> occupied(nl, 0);
         for(int ig = 0; ig < period_n[0]; ig++){
            for(int jg = 0; jg < period_n[1]; jg++){
               for(int l = 0; l < nl; l++){
                  const int k = magnetic_layers[l];
                  const int cellg = (ig*grid_n[1] + jg)*grid_n[2] + k;
                  if(global_atoms_in_cell_count[cellg] <= 0) continue;
                  offset[3*l+0] += cells_pos_and_mom_array[4*cellg+0] - (ig + 0.5)*cell_size[0];
                  offset[3*l+1] += cells_pos_and_mom_array[4*cellg+1] - (jg + 0.5)*cell_size[1];
                  offset[3*l+2] += cells_pos_and_mom_array[4*cellg+2] - (k + 0.5)*cell_size[2];
                  occupied[l]++;
               }
            }
         }
         for(int l = 0; l < nl; l++) for(int c = 0; c < 3; c++) if(occupied[l] > 0) offset[3*l+c] /= occupied[l];
         std::vector<double> position(3*period_n[0]*period_n[1]*nl);
         for(int ig = 0; ig < period_n[0]; ig++){
            for(int jg = 0; jg < period_n[1]; jg++){
               for(int l = 0; l < nl; l++){
                  const int k = magnetic_layers[l];
                  const int cellg = (ig*grid_n[1] + jg)*grid_n[2] + k;
                  double* p = &position[3*((ig*period_n[1] + jg)*nl + l)];
                  if(global_atoms_in_cell_count[cellg] > 0){
                     for(int c = 0; c < 3; c++) p[c] = cells_pos_and_mom_array[4*cellg+c];
                  }
                  else{
                     p[0] = (ig + 0.5)*cell_size[0] + offset[3*l+0];
                     p[1] = (jg + 0.5)*cell_size[1] + offset[3*l+1];
                     p[2] = (k + 0.5)*cell_size[2] + offset[3*l+2];
                  }
               }
            }
         }

         for(int lc = 0; lc < cells_num_local_cells; lc++){
            const int celli = cells_local_cell_array[lc];
            if(!field_in_cell(celli)) continue;
            const double xi = cells_pos_and_mom_array[4*celli+0];
            const double yi = cells_pos_and_mom_array[4*celli+1];
            const double zi = cells_pos_and_mom_array[4*celli+2];
            for(int l = 0; l < nl; l++){
               const int k = magnetic_layers[l];
               double* t = &uniform_tensor[6*(nl*lc + l)];
               for(int ig = 0; ig < period_n[0]; ig++){
                  for(int jg = 0; jg < period_n[1]; jg++){
                     const int cellg = (ig*grid_n[1] + jg)*grid_n[2] + k;
                     const double* p = &position[3*((ig*period_n[1] + jg)*nl + l)];
                     const int cx = nearest_copy(celli, cellg, 0), cy = nearest_copy(celli, cellg, 1);
                     for(int ix = cx - rings[0]; ix <= cx + rings[0]; ix++) for(int iy = cy - rings[1]; iy <= cy + rings[1]; iy++){
                        if(cellg == celli && ix == 0 && iy == 0) continue;
                        add_point_tensor(p[0] + ix*period[0] - xi, p[1] + iy*period[1] - yi, p[2] - zi, t);
                     }
                  }
               }
            }
         }

      }

      //------------------------------------------------------------------------
      // Function to calculate the mean moment per grid cell of every magnetic
      // layer (in Bohr magnetons), from the cell magnetisation of all cells
      //------------------------------------------------------------------------
      void layer_mean_moments(std::vector<double>& mean){
         const int nl = magnetic_layers.size();
         mean.assign(3*nl, 0.0);
         if(!periodic_images) return;
         std::vector<int> layer_index(grid_n[2], -1);
         for(int l = 0; l < nl; l++) layer_index[magnetic_layers[l]] = l;
         const double imuB = constants::i_muB;
         for(int cell = 0; cell < cells_num_cells; cell++){
            if(cells_num_atoms_in_cell[cell] <= 0) continue;
            const int l = layer_index[cell % grid_n[2]];
            if(l < 0) continue;
            mean[3*l+0] += cells::mag_array_x[cell]*imuB;
            mean[3*l+1] += cells::mag_array_y[cell]*imuB;
            mean[3*l+2] += cells::mag_array_z[cell]*imuB;
         }
         const double per_cell = 1.0/double(period_n[0]*period_n[1]);
         for(size_t c = 0; c < mean.size(); c++) mean[c] *= per_cell;
      }

      //------------------------------------------------------------------------
      // Function to replace the mean part of the layers in the summed field of
      // cell celli (local cell lc), before the field is multiplied by the dipole
      // prefactor: subtract its field over the system and the nearest copies
      // and add the field of the infinite uniform layers
      //------------------------------------------------------------------------
      void add_uniform_field(const int lc, const int celli, const std::vector<double>& mean,
                             double& bx, double& by, double& bz, double& hx, double& hy, double& hz){

         if(!periodic_images) return;
         const int nl = magnetic_layers.size();
         double fx = 0.0, fy = 0.0, fz = 0.0;
         for(int l = 0; l < nl; l++){
            const double* t = &uniform_tensor[6*(nl*lc + l)];
            const double mx = mean[3*l+0], my = mean[3*l+1], mz = mean[3*l+2];
            fx += t[0]*mx + t[1]*my + t[2]*mz;
            fy += t[1]*mx + t[3]*my + t[4]*mz;
            fz += t[2]*mx + t[4]*my + t[5]*mz;
         }
         bx -= fx; by -= fy; bz -= fz;
         hx -= fx; hy -= fy; hz -= fz;

         // inside a magnetic layer: the self term of the cell's own grid cell, and the infinite layer
         const int k = celli % grid_n[2];
         for(int l = 0; l < nl; l++){
            if(magnetic_layers[l] != k) continue;
            const double volume = cells::macro_cell_size_x*cells::macro_cell_size_y*cells::macro_cell_size_z;
            const double self = 8.0*M_PI/(3.0*volume);
            const double mx = mean[3*l+0], my = mean[3*l+1], mz = mean[3*l+2];
            bx -= self*mx; by -= self*my; bz -= self*mz;
            hx += 0.5*self*mx; hy += 0.5*self*my; hz += 0.5*self*mz;
            bx += 4.0*M_PI*mx/volume;
            by += 4.0*M_PI*my/volume;
            hz -= 4.0*M_PI*mz/volume;
         }
      }

      //------------------------------------------------------------------------
      // Function to decide whether the field of a cell is calculated
      // (dipole:field-cells = probe: cells without magnetic atoms only)
      //------------------------------------------------------------------------
      bool field_in_cell(const int cell){
         return !probe_cells_only || cells_num_atoms_in_cell[cell] == 0;
      }

   } // end of namespace internal

} // end of namespace dipole
