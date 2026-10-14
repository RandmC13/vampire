//-----------------------------------------------------------------------------
//
// This source file is part of the VAMPIRE open source package under the
// GNU GPL (version 2) licence (see licence file for details).
//
//-----------------------------------------------------------------------------

// C++ standard library headers
#include <iostream>
#include <vector>

// Vampire headers
#include "atoms.hpp"
#include "errors.hpp"
#include "hamr.hpp"
#include "material.hpp"
#include "mgpu.hpp"
#include "sim.hpp"
#include "vio.hpp"
#include "vmpi.hpp"

// CUDA module headers
#include "../cuda/cuda_utils.hpp"
#include "../cuda/data.hpp"
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

      if(cudaGetLastError() != cudaSuccess){
         std::cerr << "Error: unable to allocate multi-GPU halo buffers on rank " << vmpi::my_rank << std::endl;
         zlog << zTs() << "Error: unable to allocate multi-GPU halo buffers" << std::endl;
         err::vexit();
      }

   }

   void free_device_halo(){
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

   void pack_halo(){
      if(num_send == 0) return;
      pack_spins_kernel <<< cu::grid_size, cu::block_size >>> (d_send_atoms, num_send,
         cu::atoms::d_x_spin, cu::atoms::d_y_spin, cu::atoms::d_z_spin, d_send_buffer);
      check_cuda_errors(__FILE__, __LINE__);
      cudaMemcpy(h_send_buffer, d_send_buffer, 3 * num_send * sizeof(cu::cu_real_t), cudaMemcpyDeviceToHost);
   }

   void unpack_halo(){
      if(num_recv == 0) return;
      cudaMemcpy(d_recv_buffer, h_recv_buffer, 3 * num_recv * sizeof(cu::cu_real_t), cudaMemcpyHostToDevice);
      unpack_spins_kernel <<< cu::grid_size, cu::block_size >>> (d_recv_atoms, num_recv, d_recv_buffer,
         cu::atoms::d_x_spin, cu::atoms::d_y_spin, cu::atoms::d_z_spin);
      check_cuda_errors(__FILE__, __LINE__);
   }

} // end of internal namespace

//------------------------------------------------------------------------------
// Function to perform a single LLG Heun step: Only the local (core and boundary)
// atoms, which come first, are integrated; halo atoms only change when they are
// received, so after a step they hold the predicted spins of the last update,
// as in the CPU-only MPI version.
//------------------------------------------------------------------------------
void llg_heun(){

   // the thermal noise of the CUDA module repeats the same random numbers on every rank
   if(vmpi::num_processors > 1 && (sim::temperature > 0.0 || ::hamr::head_laser_on)){
      if(vmpi::my_rank == 0) std::cerr << "Error: finite temperatures are not yet supported by the multi-GPU version on more than one rank" << std::endl;
      zlog << zTs() << "Error: finite temperatures are not yet supported by the multi-GPU version on more than one rank" << std::endl;
      err::vexit();
   }

   if(!cu::llg::initialized) cu::llg::__llg_init();

   const size_t num_local_atoms = vmpi::num_core_atoms + vmpi::num_bdry_atoms;

   // fields at the start of the step
   internal::swap_halo();
   cu::update_spin_fields();
   cu::update_external_fields();

   cu::llg::llg_heun_predictor_step <<< cu::grid_size, cu::block_size >>> (
      cu::atoms::d_materials, cu::llg::d_heun_params,
      cu::atoms::d_x_spin, cu::atoms::d_y_spin, cu::atoms::d_z_spin,
      cu::d_x_spin_field, cu::d_y_spin_field, cu::d_z_spin_field,
      cu::d_x_external_field, cu::d_y_external_field, cu::d_z_external_field,
      cu::llg::d_ds_x, cu::llg::d_ds_y, cu::llg::d_ds_z,
      ::mp::dt, num_local_atoms);
   check_cuda_errors(__FILE__, __LINE__);

   // fields of the predicted spins
   internal::swap_halo();
   cu::update_spin_fields();

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
