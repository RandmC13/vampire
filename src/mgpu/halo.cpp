//-----------------------------------------------------------------------------
//
// This source file is part of the VAMPIRE open source package under the
// GNU GPL (version 2) licence (see licence file for details).
//
//-----------------------------------------------------------------------------

// C++ standard library headers
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

// Vampire headers
#include "mgpu.hpp"
#include "vio.hpp"
#include "vmpi.hpp"

// mgpu module headers
#include "internal.hpp"

namespace mgpu{

namespace internal{

   int num_send = 0;
   int num_recv = 0;
   void* h_send_buffer = nullptr;
   void* h_recv_buffer = nullptr;
   int real_size = 8;

   uint64_t num_swaps = 0; // number of halo updates
   double mpi_time = 0.0;  // wall time in seconds spent in MPI, including waiting for other ranks

   //---------------------------------------------------------------------------
   // Function to exchange the packed halo spins with the other ranks, with the
   // same messages as the MPI version (vmpi::mpi_init/complete_halo_swap)
   //---------------------------------------------------------------------------
   void exchange_halo(){

      const double start = MPI_Wtime();
      const MPI_Datatype type = (real_size == sizeof(double)) ? MPI_DOUBLE : MPI_FLOAT;
      char* send = static_cast<char*>(h_send_buffer);
      char* recv = static_cast<char*>(h_recv_buffer);

      std::vector<MPI_Request> requests;
      for(int p = 0; p < vmpi::num_processors; p++){
         if(vmpi::send_num_array[p] != 0){
            requests.push_back(MPI_REQUEST_NULL);
            MPI_Isend(send + 3 * vmpi::send_start_index_array[p] * real_size, 3 * vmpi::send_num_array[p], type, p, 48, MPI_COMM_WORLD, &requests.back());
         }
         if(vmpi::recv_num_array[p] != 0){
            requests.push_back(MPI_REQUEST_NULL);
            MPI_Irecv(recv + 3 * vmpi::recv_start_index_array[p] * real_size, 3 * vmpi::recv_num_array[p], type, p, 48, MPI_COMM_WORLD, &requests.back());
         }
      }
      MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
      mpi_time += MPI_Wtime() - start;

   }

   //---------------------------------------------------------------------------
   // Function to update the spins of the halo atoms on the device
   //---------------------------------------------------------------------------
   void swap_halo(){
      if(num_send == 0 && num_recv == 0) return;
      pack_halo();
      exchange_halo();
      unpack_halo();
      num_swaps++;
   }

   //---------------------------------------------------------------------------
   // Function to write the atom counts and halo exchange times of every rank.
   // The GPU is idle during the exchange, so these times add to the run time.
   //---------------------------------------------------------------------------
   void report_halo_times(){

      const int n = 6;
      const double mine[n] = {double(vmpi::num_core_atoms + vmpi::num_bdry_atoms), double(num_recv), double(num_send),
                              double(num_swaps), halo_copy_time(), mpi_time};
      std::vector<double> all(n * vmpi::num_processors);
      MPI_Gather(mine, n, MPI_DOUBLE, all.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      if(vmpi::my_rank != 0) return;

      std::ostringstream report;
      report << std::fixed << std::setprecision(3);
      for(int p = 0; p < vmpi::num_processors; p++){
         const double* r = &all[n * p];
         report << "Multi-GPU rank " << p << ": " << uint64_t(r[0]) << " local atoms, " << uint64_t(r[1]) << " halo atoms received, "
                << uint64_t(r[2]) << " sent; " << uint64_t(r[3]) << " halo updates: GPU copies " << r[4] << " s, MPI " << r[5] << " s\n";
      }
      std::cout << report.str() << std::flush;
      zlog << zTs() << "Halo exchange times\n" << report.str() << std::flush;

   }

} // end of internal namespace

bool decomposition_independent_order = false;

//------------------------------------------------------------------------------
// Function to set up halo exchange on the device and, if requested, to order
// the exchange neighbours of each atom by atom key
//------------------------------------------------------------------------------
void initialize_device(){

   if(decomposition_independent_order) internal::sort_neighbours_by_key();
   internal::initialize_noise();

   internal::num_send = vmpi::send_atom_translation_array.size();
   internal::num_recv = vmpi::recv_atom_translation_array.size();
   internal::allocate_device_halo(vmpi::send_atom_translation_array, vmpi::recv_atom_translation_array);

   zlog << zTs() << "Multi-GPU halo exchange: " << internal::num_send << " atoms sent and "
        << internal::num_recv << " atoms received per update" << std::endl;

}

//------------------------------------------------------------------------------
// Function to free buffers of the module
//------------------------------------------------------------------------------
void finalize(){
   internal::report_halo_times();
   internal::free_device_halo();
   internal::free_noise();
}

} // end of mgpu namespace
