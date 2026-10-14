//-----------------------------------------------------------------------------
//
// This source file is part of the VAMPIRE open source package under the
// GNU GPL (version 2) licence (see licence file for details).
//
//-----------------------------------------------------------------------------

// C++ standard library headers
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

// Vampire headers
#include "create.hpp"
#include "create_atoms_class.hpp"
#include "errors.hpp"
#include "mgpu.hpp"
#include "vio.hpp"
#include "vmpi.hpp"

namespace mgpu{

std::vector<uint64_t> atom_key;

//------------------------------------------------------------------------------
// Function to store a key for every atom that identifies it independently of
// the decomposition: supercell coordinates (16 bits each, wrapped into the
// system so that a periodic image has the key of the atom it copies) and the
// atom number within the unit cell (16 bits)
//------------------------------------------------------------------------------
void set_atom_keys(const std::vector<cs::catom_t>& catom_array){

   const int64_t n[3] = {int64_t(cs::total_num_unit_cells[0]), int64_t(cs::total_num_unit_cells[1]), int64_t(cs::total_num_unit_cells[2])};
   const int64_t limit = int64_t(1) << 16;

   if(n[0] >= limit || n[1] >= limit || n[2] >= limit){
      std::cerr << "Error: the multi-GPU version supports at most " << limit << " unit cells along each axis" << std::endl;
      zlog << zTs() << "Error: the multi-GPU version supports at most " << limit << " unit cells along each axis" << std::endl;
      err::vexit();
   }

   atom_key.resize(catom_array.size());
   for(size_t atom = 0; atom < catom_array.size(); atom++){
      const int64_t sc[3] = {catom_array[atom].scx, catom_array[atom].scy, catom_array[atom].scz};
      uint64_t key = catom_array[atom].uc_id;
      if(catom_array[atom].uc_id >= uint64_t(limit)){
         std::cerr << "Error: the multi-GPU version supports at most " << limit << " atoms per unit cell" << std::endl;
         zlog << zTs() << "Error: the multi-GPU version supports at most " << limit << " atoms per unit cell" << std::endl;
         err::vexit();
      }
      for(int i = 0; i < 3; i++) key |= uint64_t(((sc[i] % n[i]) + n[i]) % n[i]) << (16 * (i + 1));
      atom_key[atom] = key;
   }

   // every local (core and boundary) atom must have its own key
   std::vector<uint64_t> local(atom_key.begin(), atom_key.begin() + vmpi::num_core_atoms + vmpi::num_bdry_atoms);
   std::sort(local.begin(), local.end());
   if(std::adjacent_find(local.begin(), local.end()) != local.end()){
      std::cerr << "Error: atoms on rank " << vmpi::my_rank << " cannot be identified by their unit cell, which the multi-GPU version requires" << std::endl;
      zlog << zTs() << "Error: atoms cannot be identified by their unit cell, which the multi-GPU version requires" << std::endl;
      err::vexit();
   }

}

} // end of mgpu namespace
