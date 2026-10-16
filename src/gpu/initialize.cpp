//-----------------------------------------------------------------------------
//
// This source file is part of the VAMPIRE open source package under the
// GNU GPL (version 2) licence (see licence file for details).
//
// (c) R F L Evans 2014. All rights reserved.
//
//-----------------------------------------------------------------------------

// C++ standard library headers
#include <iostream>
#include <string>
#include <vector>

// Vampire headers
#include "anisotropy.hpp"
#include "gpu.hpp"
#include "cuda.hpp"
#include "dipole.hpp"
#include "errors.hpp"
#include "exchange.hpp"
#include "mgpu.hpp"
#include "program.hpp"
#include "sim.hpp"
#include "spintorque.hpp"
#include "spintransport.hpp"
#include "stats.hpp"
#include "vio.hpp"
#include "vmpi.hpp"
#include "vopencl.hpp"

namespace gpu{

   //-------------------------------------------------------------------------------
   // Function to stop with an error message if the simulation needs features the
   // GPU version does not calculate. Without this check such simulations run to
   // the end with these terms silently missing from the dynamics or the output.
   //
   // NOTE: this list is temporary. Remove each entry as soon as the GPU version
   // implements the feature and a validation case shows that it agrees with the
   // CPU version.
   //-------------------------------------------------------------------------------
   void check_supported_features(){

      std::vector<std::string> unsupported;

      // integrators: only llg-heun runs on the GPU. With any other integrator the host
      // spins would be overwritten by the (never updated) device spins at every
      // statistics update. Monte Carlo has a GPU version, which does not yet reproduce
      // the CPU results.
      if(sim::integrator == sim::monte_carlo) unsupported.push_back("the monte-carlo integrator");
      else if(sim::integrator != sim::llg_heun) unsupported.push_back("integrators other than llg-heun");

      // exchange
      if(::exchange::biquadratic) unsupported.push_back("biquadratic exchange");
      if(::exchange::four_spin)   unsupported.push_back("four spin exchange");

      // anisotropy
      if(anisotropy::is_triaxial_enabled())                    unsupported.push_back("triaxial anisotropy");
      if(anisotropy::is_lattice_enabled())                     unsupported.push_back("lattice anisotropy");
      if(anisotropy::is_cubic_rotation_enabled())              unsupported.push_back("rotated cubic anisotropy (cubic-anisotropy-direction)");
      if(anisotropy::is_biaxial_fourth_order_simple_enabled()) unsupported.push_back("fourth order biaxial anisotropy");
      if(sim::vcma_fields_enabled())                           unsupported.push_back("voltage controlled anisotropy");

      // fields
      if(sim::local_applied_field)        unsupported.push_back("material specific applied fields");
      if(sim::ext_demag)                  unsupported.push_back("the thin film demagnetising field (sim:demagnetisation-factor)");
      if(sim::local_temperature)          unsupported.push_back("material specific temperatures");
      if(program::program == 13)          unsupported.push_back("the localised-temperature-pulse program");
      if(::dipole::activated)             unsupported.push_back("dipole fields");
      if(sim::spin_torque_fields_enabled()) unsupported.push_back("spin transfer and spin orbit torque fields");
      if(::st::is_enabled())              unsupported.push_back("the spin torque module");
      if(::spin_transport::is_enabled())  unsupported.push_back("spin transport");
      if(sim::lagrange_multiplier)        unsupported.push_back("the Lagrange multiplier");

      // statistics
      if(::stats::calculate_system_torque || ::stats::calculate_grain_torque || ::stats::calculate_material_torque)
         unsupported.push_back("torque statistics");
      if(::stats::calculate_system_spin_temp || ::stats::calculate_grain_spin_temp || ::stats::calculate_material_spin_temp)
         unsupported.push_back("spin temperature statistics");
      if(::stats::calculate_system_lattice_temp || ::stats::calculate_grain_lattice_temp || ::stats::calculate_material_lattice_temp)
         unsupported.push_back("lattice temperature statistics");
      if(::stats::calculate_system_spin_length || ::stats::calculate_material_spin_length || ::stats::calculate_height_spin_length)
         unsupported.push_back("spin length statistics");
      if(::stats::calculate_system_binder_cumulant || ::stats::calculate_material_binder_cumulant)
         unsupported.push_back("binder cumulant statistics");
      if(::stats::calculate_system_sld_energy || ::stats::calculate_grain_sld_energy || ::stats::calculate_material_sld_energy)
         unsupported.push_back("spin-lattice energy statistics");

      if(unsupported.empty()) return;

      if(vmpi::my_rank == 0){
         terminaltextcolor(RED);
         std::cerr << "Error: GPU acceleration does not support the following features of this simulation:" << std::endl;
         for(size_t i = 0; i < unsupported.size(); i++) std::cerr << "\t" << unsupported[i] << std::endl;
         std::cerr << "Run without GPU acceleration (gpu:acceleration = false) to use them." << std::endl;
         terminaltextcolor(WHITE);
      }
      zlog << zTs() << "Error: GPU acceleration does not support the following features of this simulation:" << std::endl;
      for(size_t i = 0; i < unsupported.size(); i++) zlog << zTs() << "\t" << unsupported[i] << std::endl;
      err::vexit();

   }

   //-------------------------------------------------------------------------------
   // Function to call correct initialization function depending on cuda,opencl etc
   //-------------------------------------------------------------------------------
   void initialize(){

      // stop if the simulation needs features the GPU version does not calculate
      check_supported_features();

      // multi-GPU version: select the GPU of this rank first
      #ifdef CUDA_MULTIGPU
         mgpu::initialize();
      #endif

      #ifdef CUDA
         initialized = vcuda::initialize(gpu::cpu_stats);
      #elif OPENCL
         initialized = vopencl::initialize(gpu::cpu_stats);
      #endif

      // Check for no initialization
      if(initialized == false){
         std::cerr << "Error: no support for GPU acceleration, please recompile using -D CUDA compile" << std::endl;
         std::cerr << "time flag with cuda development libraries v6.0 or greater." << std::endl;
         std::cerr << "Aborting program." << std::endl;
         zlog << zTs() << "Error: no support for GPU acceleration, please recompile using -D CUDA compile time flag with cuda development libraries v6.0 or greater." << std::endl;
         err::vexit();
      }

      #ifdef CUDA_MULTIGPU
         mgpu::initialize_device();
      #endif

      return;
   }


   //-------------------------------------------------------------------------------
   // Function to call correct initialization function depending on cuda,opencl etc
   //-------------------------------------------------------------------------------
   void initialize_dipole(){

      // check for gpu functions and data initialised
      if( !gpu::initialized ) return;

      #ifdef CUDA
         vcuda::initialize_dipole();
      #elif OPENCL
         vopencl::initialize_dipole();
      #endif

      return;
   }


} // end of namespace gpu
