//-----------------------------------------------------------------------------
//
// This source file is part of the VAMPIRE open source package under the
// GNU GPL (version 2) licence (see licence file for details).
//
//-----------------------------------------------------------------------------

// C++ standard library headers
#include <iostream>
#include <string>
#include <vector>

// Vampire headers
#include "dipole.hpp"
#include "environment.hpp"
#include "errors.hpp"
#include "gpu.hpp"
#include "mgpu.hpp"
#include "micromagnetic.hpp"
#include "sim.hpp"
#include "spintorque.hpp"
#include "spintransport.hpp"
#include "vio.hpp"
#include "vmpi.hpp"

// mgpu module headers
#include "internal.hpp"

namespace mgpu{

namespace internal{
   int device_count(); // kernels.cu
   void device_pci_id(int device, char* id, int size);
}

//------------------------------------------------------------------------------
// Function to stop with an error message for unsupported simulations
//------------------------------------------------------------------------------
void unsupported(const std::string& feature){
   if(vmpi::my_rank == 0){
      terminaltextcolor(RED);
      std::cerr << "Error: " << feature << " is not supported by the multi-GPU version (vampire-cuda-multigpu)" << std::endl;
      terminaltextcolor(WHITE);
   }
   zlog << zTs() << "Error: " << feature << " is not supported by the multi-GPU version" << std::endl;
   err::vexit();
}

//------------------------------------------------------------------------------
// Function to select the GPU of this rank and to check that the simulation is
// supported. Ranks on the same node use the node's GPUs in turn.
//------------------------------------------------------------------------------
void initialize(){

   // simulation features handled on the host or not yet distributed
   if(vmpi::mpi_mode != 0) unsupported("MPI modes other than geometric decomposition");
   if(sim::integrator != sim::llg_heun) unsupported("Integrators other than llg-heun");
   if(::dipole::activated) unsupported("The dipole field");
   if(::st::is_enabled()) unsupported("Spin torque");
   if(::spin_transport::is_enabled()) unsupported("Spin transport");
   if(sim::lagrange_multiplier) unsupported("The Lagrange multiplier");
   if(micromagnetic::discretisation_type > 0) unsupported("Micromagnetic discretisation");
   if(environment::enabled) unsupported("The environment module");

   // rank within this node
   MPI_Comm node;
   MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
   int node_rank, node_size;
   MPI_Comm_rank(node, &node_rank);
   MPI_Comm_size(node, &node_size);

   const int num_devices = internal::device_count();
   if(num_devices < 1){
      std::cerr << "Error: no CUDA device found on rank " << vmpi::my_rank << std::endl;
      zlog << zTs() << "Error: no CUDA device found" << std::endl;
      err::vexit();
   }

   // the CUDA module sets the device chosen here (unless set in the input file)
   if(gpu::device == -1) gpu::device = node_rank % num_devices;

   // ranks may see different devices (e.g. one each), so compare PCI addresses
   const int id_size = 32;
   std::vector<char> ids(id_size * node_size, 0);
   internal::device_pci_id(gpu::device, &ids[id_size * node_rank], id_size);
   MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, ids.data(), id_size, MPI_CHAR, node);
   MPI_Comm_free(&node);
   const std::string my_id(&ids[id_size * node_rank]);
   int sharing = 0;
   for(int r = 0; r < node_size; r++) if(my_id == std::string(&ids[id_size * r])) sharing++;

   zlog << zTs() << "Rank " << vmpi::my_rank << " is rank " << node_rank << " of " << node_size
        << " on this node and uses GPU " << gpu::device << " of " << num_devices << " (PCI " << my_id << ")" << std::endl;
   if(sharing > 1){
      if(vmpi::my_rank == 0) std::cout << "Warning: rank 0 shares its GPU with " << sharing - 1 << " other rank(s)" << std::endl;
      zlog << zTs() << "Warning: this rank shares its GPU with " << sharing - 1 << " other rank(s)" << std::endl;
   }

   return;

}

} // end of mgpu namespace
