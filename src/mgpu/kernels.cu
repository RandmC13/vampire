//-----------------------------------------------------------------------------
//
// This source file is part of the VAMPIRE open source package under the
// GNU GPL (version 2) licence (see licence file for details).
//
//-----------------------------------------------------------------------------

// C++ standard library headers
#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

// Vampire headers
#include "atoms.hpp"
#include "errors.hpp"
#include "material.hpp"
#include "mgpu.hpp"
#include "vio.hpp"
#include "vmpi.hpp"

// CUDA module headers
#include "../cuda/cuda_utils.hpp"
#include "../cuda/data.hpp"
#include "../cuda/exchange_fields.hpp"
#include "../cuda/internal.hpp"
#include "../cuda/llg_heun.hpp"
#include "../cuda/typedefs.hpp"

// mgpu module headers
#include "internal.hpp"

namespace cu = ::vcuda::internal;

namespace mgpu{

namespace internal{

   // device copies of the halo atom lists and packed spins
   int* d_send_atoms = nullptr;
   int* d_recv_atoms = nullptr;
   cu::cu_real_t* d_send_buffer = nullptr;
   cu::cu_real_t* d_recv_buffer = nullptr;

   // Stream for the halo update.
   cudaStream_t halo_stream;
   cudaEvent_t step_done;

   // Events timing the halo copies on the GPU and the waiting for the default
   // stream to return the received spins; the times of one update are added after
   // the next update has been packed
   cudaEvent_t pack_start, pack_end, unpack_start, unpack_end, wait_start, wait_end, next_wait_start;
   bool unpack_pending = false;
   double copy_time = 0.0;
   double wait_time = 0.0;

   double elapsed(cudaEvent_t start, cudaEvent_t end){
      float ms = 0.0f;
      cudaEventElapsedTime(&ms, start, end);
      return 1.0e-3 * ms;
   }

   //----------------------------------------------------------------
   // Kernels to pack and unpack spins of listed atoms into buffers
   //----------------------------------------------------------------
   __global__ void pack_spins_kernel(const int* atoms, const int n,
                                     const cu::cu_real_t* x, const cu::cu_real_t* y, const cu::cu_real_t* z,
                                     cu::cu_real_t* buffer){
      for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
         const int atom = atoms[i];
         buffer[3 * i + 0] = x[atom];
         buffer[3 * i + 1] = y[atom];
         buffer[3 * i + 2] = z[atom];
      }
   }

   __global__ void unpack_spins_kernel(const int* atoms, const int n, const cu::cu_real_t* buffer,
                                       cu::cu_real_t* x, cu::cu_real_t* y, cu::cu_real_t* z){
      for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
         const int atom = atoms[i];
         x[atom] = buffer[3 * i + 0];
         y[atom] = buffer[3 * i + 1];
         z[atom] = buffer[3 * i + 2];
      }
   }

   int device_count(){
      int n = 0;
      if(cudaGetDeviceCount(&n) != cudaSuccess) return 0;
      return n;
   }

   void device_pci_id(int device, char* id, int size){
      if(cudaDeviceGetPCIBusId(id, size, device) != cudaSuccess) id[0] = '\0';
   }

   void allocate_device_halo(const std::vector<int>& send_atoms, const std::vector<int>& recv_atoms){

      real_size = sizeof(cu::cu_real_t);
      cudaMalloc((void**)&d_send_atoms, std::max<size_t>(send_atoms.size(), 1) * sizeof(int));
      cudaMalloc((void**)&d_recv_atoms, std::max<size_t>(recv_atoms.size(), 1) * sizeof(int));
      cudaMalloc((void**)&d_send_buffer, std::max<size_t>(3 * send_atoms.size(), 1) * sizeof(cu::cu_real_t));
      cudaMalloc((void**)&d_recv_buffer, std::max<size_t>(3 * recv_atoms.size(), 1) * sizeof(cu::cu_real_t));
      cudaMallocHost(&h_send_buffer, std::max<size_t>(3 * send_atoms.size(), 1) * sizeof(cu::cu_real_t));
      cudaMallocHost(&h_recv_buffer, std::max<size_t>(3 * recv_atoms.size(), 1) * sizeof(cu::cu_real_t));
      cudaMemcpy(d_send_atoms, send_atoms.data(), send_atoms.size() * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(d_recv_atoms, recv_atoms.data(), recv_atoms.size() * sizeof(int), cudaMemcpyHostToDevice);
      cudaStreamCreateWithFlags(&halo_stream, cudaStreamNonBlocking);
      cudaEventCreateWithFlags(&step_done, cudaEventDisableTiming);
      cudaEventCreate(&pack_start);
      cudaEventCreate(&pack_end);
      cudaEventCreate(&unpack_start);
      cudaEventCreate(&unpack_end);
      cudaEventCreate(&wait_start);
      cudaEventCreate(&wait_end);
      cudaEventCreate(&next_wait_start);

      if(cudaGetLastError() != cudaSuccess){
         std::cerr << "Error: unable to allocate multi-GPU halo buffers on rank " << vmpi::my_rank << std::endl;
         zlog << zTs() << "Error: unable to allocate multi-GPU halo buffers" << std::endl;
         err::vexit();
      }

   }

   void free_device_halo(){
      cudaStreamDestroy(halo_stream);
      cudaEventDestroy(step_done);
      cudaEventDestroy(pack_start);
      cudaEventDestroy(pack_end);
      cudaEventDestroy(unpack_start);
      cudaEventDestroy(unpack_end);
      cudaEventDestroy(wait_start);
      cudaEventDestroy(wait_end);
      cudaEventDestroy(next_wait_start);
      cudaFree(d_send_atoms);
      cudaFree(d_recv_atoms);
      cudaFree(d_send_buffer);
      cudaFree(d_recv_buffer);
      cudaFreeHost(h_send_buffer);
      cudaFreeHost(h_recv_buffer);
      d_send_atoms = d_recv_atoms = nullptr;
      d_send_buffer = d_recv_buffer = nullptr;
      h_send_buffer = h_recv_buffer = nullptr;
   }

   // adds the times of the last update once it has finished
   void add_pending_times(){
      if(!unpack_pending) return;
      copy_time += elapsed(unpack_start, unpack_end);
      wait_time += elapsed(wait_start, wait_end);
      unpack_pending = false;
   }

   void start_halo(){
      // spins must not be packed before the launched integration step has finished
      cudaEventRecord(step_done, 0);
      cudaStreamWaitEvent(halo_stream, step_done, 0);
      if(num_send == 0) return;
      cudaEventRecord(pack_start, halo_stream);
      pack_spins_kernel <<< cu::grid_size, cu::block_size, 0, halo_stream >>> (d_send_atoms, num_send,
         cu::atoms::d_x_spin, cu::atoms::d_y_spin, cu::atoms::d_z_spin, d_send_buffer);
      check_cuda_errors(__FILE__, __LINE__);
      cudaMemcpyAsync(h_send_buffer, d_send_buffer, 3 * num_send * sizeof(cu::cu_real_t), cudaMemcpyDeviceToHost, halo_stream);
      cudaEventRecord(pack_end, halo_stream);
   }

   void finish_halo(){
      // everything launched on the default stream so far does not need halo spins
      // (recorded in a second event, as wait_start still times the last update)
      cudaEventRecord(next_wait_start, 0);
      // the send buffer is complete and the last unpack has read the receive buffer
      cudaStreamSynchronize(halo_stream);
      if(num_send != 0) copy_time += elapsed(pack_start, pack_end);
      add_pending_times();
      std::swap(wait_start, next_wait_start);
      exchange_halo();
      if(num_recv != 0){
         cudaEventRecord(unpack_start, halo_stream);
         cudaMemcpyAsync(d_recv_buffer, h_recv_buffer, 3 * num_recv * sizeof(cu::cu_real_t), cudaMemcpyHostToDevice, halo_stream);
         unpack_spins_kernel <<< cu::grid_size, cu::block_size, 0, halo_stream >>> (d_recv_atoms, num_recv, d_recv_buffer,
            cu::atoms::d_x_spin, cu::atoms::d_y_spin, cu::atoms::d_z_spin);
         check_cuda_errors(__FILE__, __LINE__);
      }
      else cudaEventRecord(unpack_start, halo_stream);
      cudaEventRecord(unpack_end, halo_stream);
      cudaStreamWaitEvent(0, unpack_end, 0);
      cudaEventRecord(wait_end, 0);
      unpack_pending = true;
   }

   double halo_copy_time(){
      cudaDeviceSynchronize();
      add_pending_times();
      return copy_time;
   }

   double halo_wait_time(){
      cudaDeviceSynchronize();
      add_pending_times();
      return wait_time;
   }

   //---------------------------------------------------------------------------
   // Function to order the neighbours of each atom in the exchange list of the
   // CUDA module (built in local atom order) by atom key, so that the exchange
   // field of every atom is summed in the same order for any decomposition.
   // The list is only reordered: neighbours and exchange constants are unchanged.
   //---------------------------------------------------------------------------
   void sort_neighbours_by_key(){

      if(!cu::exchange::use_isotropic_kernel){
         zlog << zTs() << "Warning: exchange is not isotropic, so the order of exchange sums depends on the decomposition" << std::endl;
         return;
      }

      const int num_atoms = ::atoms::num_atoms;
      std::vector<int> start(num_atoms + 1);
      cudaMemcpy(start.data(), cu::exchange::d_nbr_start, start.size() * sizeof(int), cudaMemcpyDeviceToHost);
      const int num_nbrs = start[num_atoms];
      if(num_nbrs == 0) return;

      std::vector<int> nbr(num_nbrs);
      std::vector<cu::cu_real_t> J(num_nbrs);
      cudaMemcpy(nbr.data(), cu::exchange::d_nbr_atom, num_nbrs * sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(J.data(), cu::exchange::d_nbr_J, num_nbrs * sizeof(cu::cu_real_t), cudaMemcpyDeviceToHost);

      std::vector< std::pair<uint64_t, int> > order;
      std::vector<int> sorted_nbr(num_nbrs);
      std::vector<cu::cu_real_t> sorted_J(num_nbrs);
      for(int atom = 0; atom < num_atoms; atom++){
         order.clear();
         for(int i = start[atom]; i < start[atom + 1]; i++) order.push_back(std::make_pair(atom_key[nbr[i]], i));
         std::sort(order.begin(), order.end());
         for(size_t n = 0; n < order.size(); n++){
            sorted_nbr[start[atom] + n] = nbr[order[n].second];
            sorted_J[start[atom] + n] = J[order[n].second];
         }
      }

      cudaMemcpy(cu::exchange::d_nbr_atom, sorted_nbr.data(), num_nbrs * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(cu::exchange::d_nbr_J, sorted_J.data(), num_nbrs * sizeof(cu::cu_real_t), cudaMemcpyHostToDevice);
      zlog << zTs() << "Exchange neighbours of each atom ordered by atom key" << std::endl;

   }
   //---------------------------------------------------------------------------
   // Function to update the halo spins and the fields of the local atoms. Core
   // atoms have no halo neighbours, so their spin fields (and the external
   // fields, which need no spins) are calculated while the halo is updated, as
   // in the CPU-only MPI version; boundary atoms follow once the halo spins have
   // arrived.
   //---------------------------------------------------------------------------
   void update_fields(const bool external){

      const int num_local_atoms = vmpi::num_core_atoms + vmpi::num_bdry_atoms;

      if(!cu::exchange::use_isotropic_kernel || (num_send == 0 && num_recv == 0)){
         swap_halo();
         cu::update_spin_fields(num_local_atoms);
         if(external) update_external_fields(); // thermal noise from the atom keys (thermal.cu)
         return;
      }

      start_halo();
      cu::update_spin_fields(vmpi::num_core_atoms);
      if(external) update_external_fields();
      finish_halo();
      cu::update_spin_fields(num_local_atoms, vmpi::num_core_atoms);

   }

} // end of internal namespace

//------------------------------------------------------------------------------
// Function to perform a single LLG Heun step: Only the local (core and boundary)
// atoms, which come first, are integrated and get spin fields; halo atoms only
// change when they are received, so after a step they hold the predicted spins
// of the last update, as in the CPU-only MPI version.
//------------------------------------------------------------------------------
void llg_heun(){

   if(!cu::llg::initialized) cu::llg::__llg_init();

   const size_t num_local_atoms = vmpi::num_core_atoms + vmpi::num_bdry_atoms;

   // fields at the start of the step
   internal::update_fields(true);

   cu::llg::llg_heun_predictor_step <<< cu::grid_size, cu::block_size >>> (
      cu::atoms::d_materials, cu::llg::d_heun_params,
      cu::atoms::d_x_spin, cu::atoms::d_y_spin, cu::atoms::d_z_spin,
      cu::d_x_spin_field, cu::d_y_spin_field, cu::d_z_spin_field,
      cu::d_x_external_field, cu::d_y_external_field, cu::d_z_external_field,
      cu::llg::d_ds_x, cu::llg::d_ds_y, cu::llg::d_ds_z,
      ::mp::dt, num_local_atoms);
   check_cuda_errors(__FILE__, __LINE__);

   // fields of the predicted spins
   internal::update_fields(false);

   cu::llg::llg_heun_corrector_step <<< cu::grid_size, cu::block_size >>> (
      cu::atoms::d_materials, cu::llg::d_heun_params,
      cu::atoms::d_x_spin, cu::atoms::d_y_spin, cu::atoms::d_z_spin,
      cu::d_x_spin_field, cu::d_y_spin_field, cu::d_z_spin_field,
      cu::d_x_external_field, cu::d_y_external_field, cu::d_z_external_field,
      cu::llg::d_x_spin_buffer, cu::llg::d_y_spin_buffer, cu::llg::d_z_spin_buffer,
      cu::llg::d_ds_x, cu::llg::d_ds_y, cu::llg::d_ds_z,
      ::mp::dt, num_local_atoms);
   check_cuda_errors(__FILE__, __LINE__);

}

} // end of mgpu namespace
