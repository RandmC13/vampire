//-----------------------------------------------------------------------------
//
// This source file is part of the VAMPIRE open source package under the
// GNU GPL (version 2) licence (see licence file for details).
//
//-----------------------------------------------------------------------------
#ifndef MGPU_INTERNAL_H_
#define MGPU_INTERNAL_H_

// C++ standard library headers
#include <vector>

namespace mgpu{

   namespace internal{

      //------------------------------------------------------------------------
      // Halo exchange buffers. Spins of the boundary atoms listed in
      // vmpi::send_atom_translation_array are packed on the device into the
      // send buffer, copied to pinned host memory and sent; received spins are
      // copied to the device and unpacked into the halo atoms listed in
      // vmpi::recv_atom_translation_array. Values are stored as x,y,z triplets
      // in the precision of the CUDA build (cu_real_t).
      //------------------------------------------------------------------------
      extern int num_send;        // number of atoms sent
      extern int num_recv;        // number of atoms received
      extern void* h_send_buffer; // pinned host buffers (3 values per atom)
      extern void* h_recv_buffer;
      extern int real_size;       // size of one value in bytes (8 or 4)

      // device side (kernels.cu)
      void allocate_device_halo(const std::vector<int>& send_atoms, const std::vector<int>& recv_atoms);
      void free_device_halo();
      void pack_halo();   // device spins -> h_send_buffer
      void unpack_halo(); // h_recv_buffer -> device spins

      // host side (halo.cpp): exchanges h_send_buffer and h_recv_buffer with the other ranks
      void exchange_halo();

      // full halo update of the spins on the device
      void swap_halo();

      // orders the neighbours of each atom in the device exchange list by atom key (kernels.cu)
      void sort_neighbours_by_key();

   } // end of internal namespace

} // end of mgpu namespace

#endif //MGPU_INTERNAL_H_
