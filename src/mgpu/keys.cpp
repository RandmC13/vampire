//------------------------------------------------------------------------------
//
//   This file is part of the VAMPIRE open source package under the
//   GNU GPL (version 2) licence (see licence file for details).
//
//   (c) S Jenkins 2026. All rights reserved.
//
//------------------------------------------------------------------------------

// C++ standard library headers
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

// Vampire headers
#include "create.hpp"
#include "create_atoms_class.hpp"
#include "errors.hpp"
#include "mgpu.hpp"
#include "vio.hpp"
#include "vmpi.hpp"

namespace mgpu{

// the keys are stored by the create module (create::set_atom_keys)
std::vector<uint64_t>& atom_key = create::atom_keys;

//------------------------------------------------------------------------------
// Function to check that every atom has a key that identifies it independently
// of the decomposition: supercell coordinates (16 bits each) and the atom
// number within the unit cell (16 bits), distinct for the local atoms
//------------------------------------------------------------------------------
void set_atom_keys(const std::vector<cs::catom_t>& catom_array){

   const uint64_t limit = uint64_t(1) << 16;

   for(int i = 0; i < 3; i++){
      if(cs::total_num_unit_cells[i] >= limit){
         err::fatal("Error: the multi-GPU version supports at most " + std::to_string(limit) + " unit cells along each axis");
      }
   }
   for(size_t atom = 0; atom < catom_array.size(); atom++){
      if(catom_array[atom].uc_id >= limit){
         err::fatal("Error: the multi-GPU version supports at most " + std::to_string(limit) + " atoms per unit cell");
      }
   }

   // every local (core and boundary) atom must have its own key
   if(!create::atom_keys_valid){
      err::fatal("Error: atoms on rank " + std::to_string(vmpi::my_rank) + " cannot be identified by their unit cell, which the multi-GPU version requires");
   }

}

} // end of mgpu namespace
