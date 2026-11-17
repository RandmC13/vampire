//------------------------------------------------------------------------------
//
//   This file is part of the VAMPIRE open source package under the
//   GNU GPL (version 2) licence (see licence file for details).
//
//   (c) S Jenkins 2026. All rights reserved.
//
//------------------------------------------------------------------------------

// C++ standard library headers
#include <iostream>
#include <sstream>

// Vampire headers
#include "config.hpp"
#include "errors.hpp"
#include "gpu.hpp"
#include "hamr.hpp"
#include "sim.hpp"
#include "vio.hpp"
#include "vutil.hpp"

// hamr module headers
#include "internal.hpp"

namespace hamr{

   namespace internal{

      //------------------------------------------------------------------------
      // Function to count down the steps of a continued simulation that were
      // simulated before the checkpoint was saved; after the last of them the
      // head position and field strength of that step are restored
      //------------------------------------------------------------------------
      void skip_simulated_step(){
         resume_steps--;
         if(resume_steps == 0){
            head_position_x = resume_head_position_x;
            sim::H_applied = resume_H_applied;
            zlog << zTs() << "HAMR simulation continued from checkpoint at head position " << head_position_x*0.1 << " nm" << std::endl;
         }
      }

      //------------------------------------------------------------------------
      // Function to write the state after a track has been written
      // (hamr:save-track-states): a spin snapshot and a copy of the checkpoint,
      // vampire<rank>-track<n>.chk, which loads like any checkpoint once
      // renamed to vampire<rank>.chk
      //------------------------------------------------------------------------
      void save_track_state(const int track){
         config::output_snapshot();
         // spins on the host are not kept up to date when statistics are calculated on the GPU
         if(gpu::acceleration) gpu::transfer_spin_positions_from_gpu_to_cpu();
         std::ostringstream suffix;
         suffix << "-track" << track + 1;
         ::save_checkpoint(suffix.str());
         if(gpu::acceleration) gpu::invalidate_host_spins();
         zlog << zTs() << "State after track " << track + 1 << " saved (spin snapshot and vampire<rank>" << suffix.str() << ".chk)" << std::endl;
      }

   } // end of internal namespace

   //---------------------------------------------------------------------------
   // Function to write the state of the head to a checkpoint file: the head
   // position and field strength are accumulated step by step, so they are
   // stored rather than recalculated when the simulation is continued, followed
   // by the active window state when the window is used
   //---------------------------------------------------------------------------
   void save_checkpoint(std::ostream& chkfile){
      chkfile.write(reinterpret_cast<const char*>(&internal::head_position_x), sizeof(double));
      chkfile.write(reinterpret_cast<const char*>(&sim::H_applied), sizeof(double));
      internal::save_window_state(chkfile);
   }

   //---------------------------------------------------------------------------
   // Function to read the state of the head from a checkpoint file (older
   // checkpoint files do not contain it)
   //---------------------------------------------------------------------------
   void load_checkpoint(std::istream& chkfile){
      chkfile.read(reinterpret_cast<char*>(&internal::resume_head_position_x), sizeof(double));
      chkfile.read(reinterpret_cast<char*>(&internal::resume_H_applied), sizeof(double));
      internal::resume_state_loaded = bool(chkfile);
      if(internal::resume_state_loaded) internal::load_window_state(chkfile);
   }

} // end of hamr namespace
