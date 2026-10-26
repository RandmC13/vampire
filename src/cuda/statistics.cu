//------------------------------------------------------------------------------
//
//   This file is part of the VAMPIRE open source package under the
//   Free BSD licence (see licence file for details).
//
//   (c) R F L Evans 2015. All rights reserved.
//
//------------------------------------------------------------------------------

// C++ standard library headers
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

// Vampire headers
#include "atoms.hpp"
#include "cuda.hpp"
#include "dipole.hpp"
#include "errors.hpp"
#include "integrate.hpp"
#include "program.hpp"
#include "spintorque.hpp"
#include "spintransport.hpp"
#include "vio.hpp"

// Local cuda headers
#include "cuda_utils.hpp"
#include "internal.hpp"
#include "sim.hpp"
#include "statistics.hpp"
#include "stats.hpp"
#include "typedefs.hpp"

#ifdef CUDA
namespace cu = vcuda::internal;
#endif

namespace vcuda{

#ifdef CUDA

namespace {

   //---------------------------------------------------------------------------
   // Magnetisation statistics calculated on the device. The atoms of each
   // statistic are grouped by mask id and split into chunks of at most
   // chunk_size atoms. Each chunk is summed by one thread block in a fixed
   // order, so results are reproducible from run to run, and only the partial
   // sums of each chunk are copied to the host.
   //
   // The partial sums of each time step are kept on the device in one of
   // ring_slots slots and copied to the host in one transfer when all slots
   // are used or when the statistics are read (gpu::stats::get, reset, config
   // and checkpoint output). The host then processes the steps in order with
   // the same arithmetic as a per-step update, so the statistics are
   // unchanged, but the host does not wait for the device at every step.
   // Code reading the statistics during a simulation must call
   // gpu::stats::get() first.
   //
   // A statistic whose mask ids group the atoms of a finer
   // statistic (system and grain from material-grain, for example) is not
   // summed on the device: its sums are added from the sums of the finer one.
   //---------------------------------------------------------------------------
   const int chunk_size = 2048;
   const int reduction_block_size = 256;
   const int ring_slots = 64;

   struct device_magnetization_t{
      ::stats::magnetization_statistic_t* statistic;
      int num_chunks;
      std::vector<int> chunk_mask_id; // mask id of each chunk
      std::vector<double> h_partial;  // partial sums (mx, my, mz, m) of each chunk for each slot
      std::vector<double> sums;       // sums (mx, my, mz, m) of each mask id
      int source = -1;                // finer statistic the sums are added from (-1: summed on the device)
      std::vector<int> source_mask_id;// mask id of each mask id of the source (-1: no atoms)
      int* d_atom = nullptr;          // atom indices ordered by mask id
      double* d_moment = nullptr;     // moments of ordered atoms
      int* d_chunk_start = nullptr;   // first ordered atom of each chunk (num_chunks + 1 entries)
      double* d_partial = nullptr;    // ring_slots slots of 4 * num_chunks partial sums
   };

   std::vector<device_magnetization_t> device_statistics;
   bool device_statistics_initialised = false;
   bool fallback_logged = false;
   int pending_steps = 0; // steps summed on the device but not yet processed on the host

   //---------------------------------------------------------------------------
   // Function to stop if a kernel has failed. Release builds do not check each
   // kernel, so this is called after copies from the device, which wait for all
   // earlier kernels and so report their errors without an extra synchronisation.
   //---------------------------------------------------------------------------
   void check_device_errors(){
      const cudaError_t error = cudaGetLastError();
      if(error != cudaSuccess){
         ::err::fatal(std::string("Error: CUDA error on the GPU: ") + cudaGetErrorString(error));
      }
   }

   //---------------------------------------------------------------------------
   // Kernel summing spin moments of each chunk
   //---------------------------------------------------------------------------
   __global__ void sum_magnetization_kernel(const int* atom, const double* moment, const int* chunk_start,
                                            const cu::cu_real_t* sx, const cu::cu_real_t* sy, const cu::cu_real_t* sz,
                                            double* partial){

      __shared__ double sum[4][reduction_block_size];

      const int t = threadIdx.x;
      double mx = 0.0, my = 0.0, mz = 0.0, m = 0.0;
      for(int i = chunk_start[blockIdx.x] + t; i < chunk_start[blockIdx.x + 1]; i += reduction_block_size){
         const int a = atom[i];
         const double mm = moment[i];
         mx += sx[a] * mm;
         my += sy[a] * mm;
         mz += sz[a] * mm;
         m  += mm;
      }
      sum[0][t] = mx;
      sum[1][t] = my;
      sum[2][t] = mz;
      sum[3][t] = m;
      __syncthreads();

      // fixed order tree reduction
      for(int stride = reduction_block_size / 2; stride > 0; stride /= 2){
         if(t < stride){
            for(int k = 0; k < 4; k++) sum[k][t] += sum[k][t + stride];
         }
         __syncthreads();
      }

      if(t < 4) partial[4 * blockIdx.x + t] = sum[t][0];

   }

   //---------------------------------------------------------------------------
   // Function to find the mask id of a statistic for each mask id of a finer
   // statistic; false if there is a discrepancy in the mask ids for the atoms
   // between the two statistics
   //---------------------------------------------------------------------------
   bool nested_mask(const std::vector<int>& fine, const int fine_size, const std::vector<int>& coarse, std::vector<int>& fine_to_coarse){
      if(fine.size() != coarse.size()) return false;
      fine_to_coarse.assign(fine_size, -1);
      for(size_t atom = 0; atom < fine.size(); atom++){
         int& id = fine_to_coarse[fine[atom]];
         if(id == -1) id = coarse[atom];
         else if(id != coarse[atom]) return false;
      }
      return true;
   }

   //---------------------------------------------------------------------------
   // Function to set up device data for one magnetisation statistic
   //---------------------------------------------------------------------------
   void add_device_statistic(device_magnetization_t& dm, const std::vector<int>& mask, const int num_mask_ids){

      const int num_atoms = mask.size();
      dm.sums.resize(4 * num_mask_ids);

      // order atoms by mask id (counting sort, keeps atom order within each mask id)
      std::vector<int> mask_start(num_mask_ids + 1, 0);
      for(int atom = 0; atom < num_atoms; atom++) mask_start[mask[atom] + 1]++;
      for(int id = 0; id < num_mask_ids; id++) mask_start[id + 1] += mask_start[id];
      std::vector<int> position(mask_start.begin(), mask_start.end() - 1);
      std::vector<int> atom_order(num_atoms);
      std::vector<double> moment(num_atoms);
      for(int atom = 0; atom < num_atoms; atom++){
         const int i = position[mask[atom]]++;
         atom_order[i] = atom;
         moment[i] = ::atoms::m_spin_array[atom];
      }

      // split each mask id into chunks
      std::vector<int> chunk_start;
      for(int id = 0; id < num_mask_ids; id++){
         for(int i = mask_start[id]; i < mask_start[id + 1]; i += chunk_size){
            chunk_start.push_back(i);
            dm.chunk_mask_id.push_back(id);
         }
      }
      chunk_start.push_back(num_atoms);
      dm.num_chunks = dm.chunk_mask_id.size();
      dm.h_partial.resize(ring_slots * 4 * dm.num_chunks);

      cudaMalloc((void**)&dm.d_atom, num_atoms * sizeof(int));
      cudaMalloc((void**)&dm.d_moment, num_atoms * sizeof(double));
      cudaMalloc((void**)&dm.d_chunk_start, chunk_start.size() * sizeof(int));
      cudaMalloc((void**)&dm.d_partial, dm.h_partial.size() * sizeof(double));
      cudaMemcpy(dm.d_atom, atom_order.data(), num_atoms * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(dm.d_moment, moment.data(), num_atoms * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(dm.d_chunk_start, chunk_start.data(), chunk_start.size() * sizeof(int), cudaMemcpyHostToDevice);

      check_cuda_errors(__FILE__, __LINE__);

   }

   //---------------------------------------------------------------------------
   // Function to set up the magnetisation statistics calculated on the device
   //---------------------------------------------------------------------------
   void initialise_device_statistics(){

      // the host processes the statistics in this order, the same on every process
      std::vector<::stats::magnetization_statistic_t*> list;
      if(::stats::calculate_system_magnetization)          list.push_back(&::stats::system_magnetization);
      if(::stats::calculate_material_magnetization)        list.push_back(&::stats::material_magnetization);
      if(::stats::calculate_height_magnetization)          list.push_back(&::stats::height_magnetization);
      if(::stats::calculate_material_height_magnetization) list.push_back(&::stats::material_height_magnetization);
      if(::stats::calculate_grain_magnetization)           list.push_back(&::stats::grain_magnetization);
      if(::stats::calculate_material_grain_magnetization)  list.push_back(&::stats::material_grain_magnetization);
      if(::stats::calculate_material_grain_height_magnetization) list.push_back(&::stats::material_grain_height_magnetization);

      const int n = list.size();
      std::vector<std::vector<int>> masks(n);
      std::vector<int> sizes(n);
      device_statistics.resize(n);
      for(int i = 0; i < n; i++){
         std::vector<double> saturation;
         list[i]->get_mask(masks[i], saturation);
         sizes[i] = saturation.size();
         device_statistics[i].statistic = list[i];
      }

      // finest statistics first; each other one is added from the coarsest summed statistic it groups
      std::vector<int> order(n);
      for(int i = 0; i < n; i++) order[i] = i;
      std::stable_sort(order.begin(), order.end(), [&](int a, int b){ return sizes[a] > sizes[b]; });
      std::vector<int> summed;
      std::vector<int> fine_to_coarse;
      for(const int i : order){
         device_magnetization_t& dm = device_statistics[i];
         for(const int j : summed){
            if((dm.source == -1 || sizes[j] < sizes[dm.source]) && nested_mask(masks[j], sizes[j], masks[i], fine_to_coarse)){
               dm.source = j;
               dm.source_mask_id = fine_to_coarse;
            }
         }
         if(dm.source == -1){
            add_device_statistic(dm, masks[i], sizes[i]);
            summed.push_back(i);
         }
         else{
            dm.num_chunks = 0;
            dm.sums.resize(4 * sizes[i]);
         }
      }
      // This line might need better wording
      zlog << zTs() << "Magnetisation statistics: " << summed.size() << " of " << n << " summed on the GPU, the others added from their sums" << std::endl;

   }

   //---------------------------------------------------------------------------
   // Function to determine whether the program may read the spins on the host
   // directly during the simulation
   //---------------------------------------------------------------------------
   bool program_reads_host_spins(){

      // programs which only use statistics of the spins during the simulation
      switch(::program::program){
         case ::program::program_t::time_series:
         case ::program::program_t::hysteresis_loop:
         case ::program::program_t::curie_temperature:
         case ::program::program_t::field_cool:
         case ::program::program_t::laser_pulse:
         case ::program::program_t::hamr_simulation:
         case ::program::program_t::partial_hysteresis_loop:
         case ::program::program_t::fmr:
            return false;
         default:
            return true;
      }

   }

   //---------------------------------------------------------------------------
   // Function to determine whether the host needs an up to date copy of the
   // spins at every statistics update, in which case all statistics are
   // calculated on the CPU
   //---------------------------------------------------------------------------
   bool host_spins_required(){

      // statistics needing the spins themselves
      if(::stats::calculate_system_energy || ::stats::calculate_grain_energy || ::stats::calculate_material_energy) return true;

      // modules reading the spins on the host every time step
      if(::dipole::activated || ::st::is_enabled() || ::spin_transport::is_enabled() || ::program::lagrange_multiplier_enabled) return true;

      // only the GPU integrators keep the spins on the device up to date
      if(::integrate::integrator != ::integrate::integrator_t::llg_heun && ::integrate::integrator != ::integrate::integrator_t::monte_carlo) return true;

      return program_reads_host_spins();

   }

   //---------------------------------------------------------------------------
   // Function to update statistics derived from the magnetisation and energy
   //---------------------------------------------------------------------------
   void update_derived_statistics(){

      // update torque statistics
      //if(stats::calculate_system_torque)          stats::system_torque.calculate_torque(sx,sy,sz,bxs,bys,bzs,bxe,bye,bze,mm);
      //if(stats::calculate_grain_torque)           stats::grain_torque.calculate_torque(sx,sy,sz,bxs,bys,bzs,bxe,bye,bze,mm);
      //if(stats::calculate_material_torque)        stats::material_torque.calculate_torque(sx,sy,sz,bxs,bys,bzs,bxe,bye,bze,mm);

      // update specific heat statistics
      if(::stats::calculate_system_specific_heat)         ::stats::system_specific_heat.calculate(::stats::system_energy.get_total_energy());
      if(::stats::calculate_grain_specific_heat)          ::stats::grain_specific_heat.calculate(::stats::grain_energy.get_total_energy());
      if(::stats::calculate_material_specific_heat)       ::stats::material_specific_heat.calculate(::stats::material_energy.get_total_energy());

      // standard deviation in time-step
      if(::stats::calculate_material_standard_deviation)  ::stats::material_standard_deviation.update(::stats::system_magnetization.get_magnetization());

      // update susceptibility statistics
      if(::stats::calculate_system_susceptibility)        ::stats::system_susceptibility.calculate(::stats::system_magnetization.get_magnetization());
      if(::stats::calculate_grain_susceptibility)         ::stats::grain_susceptibility.calculate(::stats::grain_magnetization.get_magnetization());
      if(::stats::calculate_material_susceptibility)      ::stats::material_susceptibility.calculate(::stats::material_magnetization.get_magnetization());

   }

   //---------------------------------------------------------------------------
   // Function to process, in order, the steps summed on the device since the
   // last call, as a per-step update would have done
   //---------------------------------------------------------------------------
   void process_pending_steps(){

      if(pending_steps == 0) return;

      for(device_magnetization_t& dm : device_statistics){
         if(dm.num_chunks > 0) cudaMemcpy(dm.h_partial.data(), dm.d_partial, pending_steps * 4 * dm.num_chunks * sizeof(double), cudaMemcpyDeviceToHost);
      }
      check_device_errors();

      for(int step = 0; step < pending_steps; step++){
         for(device_magnetization_t& dm : device_statistics){
            if(dm.source != -1) continue;
            // add chunks in a fixed order
            const double* partial = dm.h_partial.data() + step * 4 * dm.num_chunks;
            std::fill(dm.sums.begin(), dm.sums.end(), 0.0);
            for(int c = 0; c < dm.num_chunks; c++){
               for(int k = 0; k < 4; k++) dm.sums[4 * dm.chunk_mask_id[c] + k] += partial[4 * c + k];
            }
         }
         for(device_magnetization_t& dm : device_statistics){
            if(dm.source == -1) continue;
            // add the sums of the finer statistic in a fixed order
            const std::vector<double>& fine = device_statistics[dm.source].sums;
            std::fill(dm.sums.begin(), dm.sums.end(), 0.0);
            for(size_t f = 0; f < dm.source_mask_id.size(); f++){
               const int id = dm.source_mask_id[f];
               if(id != -1) for(int k = 0; k < 4; k++) dm.sums[4 * id + k] += fine[4 * f + k];
            }
         }
         for(device_magnetization_t& dm : device_statistics) dm.statistic->calculate_magnetization_from_sums(dm.sums);
         update_derived_statistics();
      }

      pending_steps = 0;

   }

   //---------------------------------------------------------------------------
   // Function to sum the magnetisation of the current step on the device
   //---------------------------------------------------------------------------
   void sum_device_magnetization(){

      if(!device_statistics_initialised){
         initialise_device_statistics();
         device_statistics_initialised = true;
      }

      for(device_magnetization_t& dm : device_statistics){
         if(dm.num_chunks > 0){
            sum_magnetization_kernel <<< dm.num_chunks, reduction_block_size >>> (
               dm.d_atom, dm.d_moment, dm.d_chunk_start,
               cu::atoms::d_x_spin, cu::atoms::d_y_spin, cu::atoms::d_z_spin,
               dm.d_partial + pending_steps * 4 * dm.num_chunks);
            check_cuda_errors(__FILE__, __LINE__);
         }
      }
      pending_steps++;

      if(pending_steps == ring_slots) process_pending_steps();

   }

} // end of anonymous namespace

//------------------------------------------------------------------------------
// Function to copy the spins to the host after integration if the program may
// read them there directly (programs that do not use the GPU statistics)
//------------------------------------------------------------------------------
void update_host_spins(){
   if(program_reads_host_spins()) transfer_spin_positions_from_gpu_to_cpu();
}

namespace stats{
   void update (){

      // Calculate statistics on the CPU by copying the spins from the GPU if requested,
      // or if the host needs an up to date copy of the spins anyway
      const bool host_spins = host_spins_required();
      if(vcuda::internal::stats::use_cpu || host_spins){

         if(host_spins && !vcuda::internal::stats::use_cpu && !fallback_logged){
            zlog << zTs() << "Statistics are calculated on the CPU since the spins are needed on the host at every update" << std::endl;
            fallback_logged = true;
         }

         cudaMemcpy(internal::h_x_spin_transfer_buffer, internal::atoms::d_x_spin, ::atoms::num_atoms * sizeof(cu::cu_real_t), cudaMemcpyDeviceToHost);
         cudaMemcpy(internal::h_y_spin_transfer_buffer, internal::atoms::d_y_spin, ::atoms::num_atoms * sizeof(cu::cu_real_t), cudaMemcpyDeviceToHost);
         cudaMemcpy(internal::h_z_spin_transfer_buffer, internal::atoms::d_z_spin, ::atoms::num_atoms * sizeof(cu::cu_real_t), cudaMemcpyDeviceToHost);
         check_device_errors();

         std::copy(internal::h_x_spin_transfer_buffer, internal::h_x_spin_transfer_buffer + ::atoms::num_atoms, ::atoms::x_spin_array.begin());
         std::copy(internal::h_y_spin_transfer_buffer, internal::h_y_spin_transfer_buffer + ::atoms::num_atoms, ::atoms::y_spin_array.begin());
         std::copy(internal::h_z_spin_transfer_buffer, internal::h_z_spin_transfer_buffer + ::atoms::num_atoms, ::atoms::z_spin_array.begin());
         cu::stats::device_owns_spins = false;
         cu::stats::host_spins_current = true;

         //---------------------------------------------------------------
         // call cpu statistics functions
         //---------------------------------------------------------------

         // update energy statistics
         if(::stats::calculate_system_energy) ::stats::system_energy.calculate(::atoms::x_spin_array, ::atoms::y_spin_array, ::atoms::z_spin_array, ::atoms::m_spin_array, ::atoms::type_array , ::sim::temperature);
         if (::stats::calculate_grain_energy) ::stats::grain_energy.calculate(::atoms::x_spin_array, ::atoms::y_spin_array, ::atoms::z_spin_array, ::atoms::m_spin_array, ::atoms::type_array, ::sim::temperature);
         if(::stats::calculate_material_energy) ::stats::material_energy.calculate(::atoms::x_spin_array, ::atoms::y_spin_array, ::atoms::z_spin_array, ::atoms::m_spin_array, ::atoms::type_array, ::sim::temperature);

         // update magnetization statistics
         if(::stats::calculate_system_magnetization)          ::stats::system_magnetization.calculate_magnetization(::atoms::x_spin_array, ::atoms::y_spin_array, ::atoms::z_spin_array, ::atoms::m_spin_array);
         if(::stats::calculate_material_magnetization)        ::stats::material_magnetization.calculate_magnetization(::atoms::x_spin_array, ::atoms::y_spin_array, ::atoms::z_spin_array, ::atoms::m_spin_array);
         if(::stats::calculate_height_magnetization)          ::stats::height_magnetization.calculate_magnetization(::atoms::x_spin_array, ::atoms::y_spin_array, ::atoms::z_spin_array, ::atoms::m_spin_array);
         if(::stats::calculate_material_height_magnetization) ::stats::material_height_magnetization.calculate_magnetization(::atoms::x_spin_array, ::atoms::y_spin_array, ::atoms::z_spin_array, ::atoms::m_spin_array);
         if(::stats::calculate_grain_magnetization)           ::stats::grain_magnetization.calculate_magnetization(::atoms::x_spin_array, ::atoms::y_spin_array, ::atoms::z_spin_array, ::atoms::m_spin_array);
         if(::stats::calculate_material_grain_magnetization)  ::stats::material_grain_magnetization.calculate_magnetization(::atoms::x_spin_array, ::atoms::y_spin_array, ::atoms::z_spin_array, ::atoms::m_spin_array);
         if(::stats::calculate_material_grain_height_magnetization) ::stats::material_grain_height_magnetization.calculate_magnetization(::atoms::x_spin_array, ::atoms::y_spin_array, ::atoms::z_spin_array, ::atoms::m_spin_array);

         // statistics derived from the magnetisation and energy
         update_derived_statistics();

      }
      else{
         sum_device_magnetization();
         // the spins on the host are not updated from now on
         cu::stats::device_owns_spins = true;
         vcuda::invalidate_host_spins();
      }

      // increase the counter
      cu::stats::counter++;

   }

   void get (){

      // process steps summed on the device but not yet on the host
      process_pending_steps();

   }

   void reset (){

      // steps before the reset must not be counted after it
      process_pending_steps();

      // reset statistics (calculated or accumulated on the CPU for both CPU and GPU statistics)
      {

         // reset energy statistics
         if(::stats::calculate_system_energy)                 ::stats::system_energy.reset_averages();
         if(::stats::calculate_grain_energy)                  ::stats::grain_energy.reset_averages();
         if(::stats::calculate_material_energy)               ::stats::material_energy.reset_averages();

         // reset magnetization statistics
         if(::stats::calculate_system_magnetization)          ::stats::system_magnetization.reset_magnetization_averages();
         if(::stats::calculate_grain_magnetization)           ::stats::grain_magnetization.reset_magnetization_averages();
         if(::stats::calculate_material_magnetization)        ::stats::material_magnetization.reset_magnetization_averages();
         if(::stats::calculate_material_grain_magnetization)  ::stats::material_grain_magnetization.reset_magnetization_averages();
         if(::stats::calculate_height_magnetization)          ::stats::height_magnetization.reset_magnetization_averages();
         if(::stats::calculate_material_height_magnetization) ::stats::material_height_magnetization.reset_magnetization_averages();
         if(::stats::calculate_material_grain_height_magnetization) ::stats::material_grain_height_magnetization.reset_magnetization_averages();

         // update torque statistics
         //if(stats::calculate_system_torque)          stats::system_torque.reset_torque_averages();
         //if(stats::calculate_grain_torque)           stats::grain_torque.reset_torque_averages();
         //if(stats::calculate_material_torque)        stats::material_torque.reset_torque_averages();

         // standard deviation in time-step
         if(::stats::calculate_material_standard_deviation)     ::stats::material_standard_deviation.reset_averages();

         // reset specific_heat statistics
         if(::stats::calculate_system_specific_heat)   ::stats::system_specific_heat.reset_averages();
         if(::stats::calculate_grain_specific_heat)    ::stats::grain_specific_heat.reset_averages();
         if(::stats::calculate_material_specific_heat) ::stats::material_specific_heat.reset_averages();

         // reset susceptibility statistics
         if(::stats::calculate_system_susceptibility)   ::stats::system_susceptibility.reset_averages();
         if(::stats::calculate_grain_susceptibility)    ::stats::grain_susceptibility.reset_averages();
         if(::stats::calculate_material_susceptibility) ::stats::material_susceptibility.reset_averages();

         // reset binder cumulant statistics
         //if(stats::calculate_system_binder_cumulant)   stats::system_binder_cumulant.reset_averages();
         //if(stats::calculate_material_binder_cumulant) stats::material_binder_cumulant.reset_averages();

         return;
      }
   }

} /* stats */

   namespace internal
   {
      namespace stats
      {
         void finalise(){
            process_pending_steps();
            for(device_magnetization_t& dm : device_statistics){
               cudaFree(dm.d_atom);
               cudaFree(dm.d_moment);
               cudaFree(dm.d_chunk_start);
               cudaFree(dm.d_partial);
            }
            device_statistics.clear();
            pending_steps = 0;
            device_statistics_initialised = false;
         }
      } /* stats */
   } /* internal */

#endif

} // end of namespace cuda
