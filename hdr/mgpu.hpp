//-----------------------------------------------------------------------------
//
// This source file is part of the VAMPIRE open source package under the
// GNU GPL (version 2) licence (see licence file for details).
//
//-----------------------------------------------------------------------------
#ifndef MGPU_H_
#define MGPU_H_

//-----------------------------------------------------------------------------
// Multi-GPU module (make cuda-multigpu): one MPI rank per GPU. The system is
// decomposed geometrically as in the MPI version; each rank integrates its
// core and boundary atoms on its GPU with the kernels of the CUDA module and
// exchanges the spins of halo atoms with its neighbours before every field
// evaluation.
//-----------------------------------------------------------------------------
namespace mgpu{

   // Selects the GPU of this rank and checks that the simulation is supported.
   // Called before the CUDA module is initialised.
   void initialize();

   // Sets up halo exchange on the device. Called after the CUDA module is initialised.
   void initialize_halo();

   // Performs a single LLG Heun step on the device
   void llg_heun();

   // Frees device and host buffers of the module
   void finalize();

} // end of mgpu namespace

#endif //MGPU_H_
