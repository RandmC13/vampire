//-----------------------------------------------------------------------------
//
// This source file is part of the VAMPIRE open source package under the
// GNU GPL (version 2) licence (see licence file for details).
//
//-----------------------------------------------------------------------------

// C++ standard library headers
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

   //---------------------------------------------------------------------------
   // Function to exchange the packed halo spins with the other ranks, with the
   // same messages as the MPI version (vmpi::mpi_init/complete_halo_swap)
   //---------------------------------------------------------------------------
   void exchange_halo(){

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

   }

   //---------------------------------------------------------------------------
   // Function to update the spins of the halo atoms on the device
   //---------------------------------------------------------------------------
   void swap_halo(){
      if(num_send == 0 && num_recv == 0) return;
      pack_halo();
      exchange_halo();
      unpack_halo();
   }

} // end of internal namespace

//------------------------------------------------------------------------------
// Function to set up halo exchange on the device
//------------------------------------------------------------------------------
void initialize_halo(){

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
   internal::free_device_halo();
}

} // end of mgpu namespace
