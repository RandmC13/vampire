//------------------------------------------------------------------------------
//
//   This file is part of the VAMPIRE open source package under the
//   Free BSD licence (see licence file for details).
//
//   (c) Sarah Jenkins and Richard F L Evans 2019. All rights reserved.
//
//------------------------------------------------------------------------------
//

// C++ standard library headers
#include <cmath>
#include <vector>

// Vampire headers
#include "cells.hpp"
#include "hierarchical.hpp"

// hierarchical module headers
#include "internal.hpp"

namespace ha = hierarchical::internal;

namespace hierarchical{

   namespace internal{

      //------------------------------------------------------------------------
      // Appends to interaction_list the cells that local cell lc interacts
      // with, from the coarsest level down: a coarse cell all of whose corners
      // lie outside the range of its level stands for all the cells it
      // contains; the remaining zero-level cells interact individually. The
      // interactions are interaction_list[start_index[slot] .. end_index[slot]).
      //------------------------------------------------------------------------
      void add_cell_interactions(const int lc, const int slot){

         std::vector < bool > interacted(ha::total_num_cells,false);
         ha::interaction_list_start_index[slot] = ha::interaction_list.size();

         const int cell_i = cells::cell_id_array[lc];
         const double xi = ha::cell_positions[cell_i*3 + 0];
         const double yi = ha::cell_positions[cell_i*3 + 1];
         const double zi = ha::cell_positions[cell_i*3 + 2];

         for (int level = ha::num_levels - 1; level > -1; level -- ){

            const int start = ha::cells_level_start_index[level];
            const int end   = ha::cells_level_end_index[level];
            for (int cell_j = start; cell_j < end; cell_j++){

               if (interacted[cell_j] == false && level != 0){

                  const std::vector < std::vector < double> > corners = ha::calculate_corners(ha::cell_positions[cell_j*3 + 0], ha::cell_positions[cell_j*3 + 1], ha::cell_positions[cell_j*3 + 2],
                                                                                            ha::cell_dimensions[cell_j*3 + 0], ha::cell_dimensions[cell_j*3 + 1], ha::cell_dimensions[cell_j*3 + 2]);

                  int is_corner_outside_range = 0;
                  for (int corner = 0; corner < 8; corner ++){
                     const double dx_2 = (xi - corners[corner][0])*(xi-corners[corner][0]);
                     const double dy_2 = (yi - corners[corner][1])*(yi-corners[corner][1]);
                     const double dz_2 = (zi - corners[corner][2])*(zi-corners[corner][2]);
                     const double r = sqrt(dx_2 + dy_2 + dz_2);
                     if (r > ha::interaction_range[level]) is_corner_outside_range ++;
                  }

                  // whole cell out of range: interact with it, and not with the cells it contains
                  if (is_corner_outside_range == 8){
                     interacted[cell_j] = true;
                     ha::interaction_list.push_back(cell_j);
                     for (int interaction = ha::cells_in_cells_start_index[cell_j]; interaction < ha::cells_in_cells_end_index[cell_j]; interaction ++ ){
                        interacted[ha::cells_in_cells[interaction]] = true;
                     }
                  }
               }
               else if (level != 0){
                  for (int interaction = ha::cells_in_cells_start_index[cell_j]; interaction < ha::cells_in_cells_end_index[cell_j]; interaction ++ ){
                     interacted[ha::cells_in_cells[interaction]] = true;
                  }
               }
               else if (level == 0 && interacted[cell_j] == false){
                  ha::interaction_list.push_back(cell_j);
               }
            }
         }

         ha::interaction_list_end_index[slot] = ha::interaction_list.size();

         return;

      }

      //------------------------------------------------------------------------
      // Computes the dipole tensors of the interactions of local cell lc
      // listed for slot, into rij_tensor_*[interaction]. Zero-level cells
      // within the cutoff use the atomistic inter/intra tensors; other
      // zero-level cells and coarse cells are point dipoles at their centres
      // of moment. Cells without magnetic atoms are no sources (zero tensor).
      //------------------------------------------------------------------------
      void compute_cell_tensors(const int lc, const int slot){

         const int cell_i = cells::cell_id_array[lc];

         for(int j = ha::interaction_list_start_index[slot]; j < ha::interaction_list_end_index[slot]; j++){

            const int cell_j = ha::interaction_list[j];

            ha::rij_tensor_xx[j] = 0.0;
            ha::rij_tensor_xy[j] = 0.0;
            ha::rij_tensor_xz[j] = 0.0;
            ha::rij_tensor_yy[j] = 0.0;
            ha::rij_tensor_yz[j] = 0.0;
            ha::rij_tensor_zz[j] = 0.0;

            if(cell_j < ha::num_zero_level_cells){

               if(ha::tensor_atoms_in_cell_global[cell_j] == 0) continue;

               if(cell_i != cell_j){
                  internal::calc_inter(cell_i, cell_j, j, ha::tensor_cutoff, ha::tensor_atoms_in_cell_global, ha::tensor_cells_pos_and_mom,
                                       ha::tensor_list_of_cells_with_atoms, ha::tensor_atoms_in_cells);
               }
               else{
                  internal::calc_intra(cell_i, cell_j, j, ha::tensor_atoms_in_cell_global, ha::tensor_list_of_cells_with_atoms, ha::tensor_atoms_in_cells);
               }

            }
            // coarse cell: point dipole at its centre of moment (fourth entry: total moment, zero without magnetic atoms)
            else if(ha::cell_positions_mom[4*cell_j+3] > 0.0){

               const double rx = ha::cell_positions_mom[4*cell_j+0] - ha::cell_positions_mom[4*cell_i+0];
               const double ry = ha::cell_positions_mom[4*cell_j+1] - ha::cell_positions_mom[4*cell_i+1];
               const double rz = ha::cell_positions_mom[4*cell_j+2] - ha::cell_positions_mom[4*cell_i+2];

               const double rij = 1.0/sqrt(rx*rx + ry*ry + rz*rz);
               const double ex = rx*rij;
               const double ey = ry*rij;
               const double ez = rz*rij;
               const double rij3 = (rij*rij*rij); // Angstroms

               ha::rij_tensor_xx[j] = ((3.0*ex*ex - 1.0)*rij3);
               ha::rij_tensor_xy[j] = ((3.0*ex*ey      )*rij3);
               ha::rij_tensor_xz[j] = ((3.0*ex*ez      )*rij3);
               ha::rij_tensor_yy[j] = ((3.0*ey*ey - 1.0)*rij3);
               ha::rij_tensor_yz[j] = ((3.0*ey*ez      )*rij3);
               ha::rij_tensor_zz[j] = ((3.0*ez*ez - 1.0)*rij3);

            }
         }

         return;

      }

   } // end of namespace internal

} // end of namespace hierarchical
