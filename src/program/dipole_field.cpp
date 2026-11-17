//------------------------------------------------------------------------------
//
//   This file is part of the VAMPIRE open source package under the
//   GNU GPL (version 2) licence (see licence file for details).
//
//   (c) S Jenkins 2026. All rights reserved.
//
//------------------------------------------------------------------------------
//

// Standard Libraries
#include <iostream>

// Vampire Header files
#include "dipole.hpp"
#include "errors.hpp"
#include "gpu.hpp"
#include "program.hpp"
#include "sim.hpp"
#include "stats.hpp"
#include "vio.hpp"

namespace program{

//------------------------------------------------------------------------------
// Program to calculate the dipole field of the current spin configuration once,
// usually loaded from a checkpoint (sim:load-checkpoint), without integrating:
// the field is calculated and written (dipole:output-dipole-field,
// dipole:output-atomistic-dipole-field) when the dipole module is initialised,
// and the statistics of the configuration are written once
//------------------------------------------------------------------------------
void dipole_field(){

   // check calling of routine if error checking is activated
   if(err::check==true) std::cout << "program::dipole_field has been called" << std::endl;

   if(!dipole::activated) err::fatal("Error - sim:program = dipole-field needs a dipole solver (dipole:solver).");
   if(gpu::acceleration) err::fatal("Error - sim:program = dipole-field is only available in the CPU versions.");

   stats::update();
   vout::data();

}

} // end of namespace program
