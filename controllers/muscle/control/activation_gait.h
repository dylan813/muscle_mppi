#pragma once

// Activation gait generation for the muscle variant.
//
// An activation gait is the muscle-space version of a joint-space gait table:
// for every phase column of the table (joint positions + velocities,
// gait_gen/gaits/<label>.tsv, see gait_table_path() in common/gait.h), each
// joint's activation pair is the point on its torque-balance line at
// MuscleParams::stiffness that produces the joint's loaded torque: gravity +
// Coriolis less the planned stance forces (gait_forces_path(), stance_load.h),
// body still and level — hill_invert_torque().
//
// Rows: 0..11 a1 (agonist), 12..23 a2 (antagonist), 24..35 those planned
// forces, copied, so the gait-tracking cost loads them in phase with the
// activations (GaitScheduler, gait_scheduler.h).
//
// Generated files live in gait_gen/activation_gaits/ (gitignored) and start with
// one header line recording what they were generated from:
//   # activation_gait source=<label> table=<fingerprint> stiffness=<value> muscle=<fingerprint>
// table= covers the table and its forces file. The gait loader skips the line
// (non-numeric), as does np.loadtxt ('#'). ensure_activation_gait() regenerates
// a file whenever that line doesn't match the current table and forces (label
// and content) and muscle parameters (including stiffness).

#include <string>

#include "../utils/tasks.h"

// Repo-relative path of the generated activation gait for a gait label, e.g.
// "walk" -> gait_gen/activation_gaits/walk.tsv.
std::string activation_gait_path(const std::string& label);

// Hash of every muscle parameter the generated gait depends on (stiffness,
// lce_min/max, phi_min/max, pFLmax, FVmax, vmax, peak_force, and the Hill
// ablation switches when any is off) plus a generator version, as 16 hex digits.
std::string muscle_fingerprint(const MuscleParams& p);

// The header line (without "# " and newline) a gait generated from the table
// labelled `label`, whose content and forces hash to `table`, with `p` carries.
std::string activation_gait_header(const std::string& label, const std::string& table,
                                   const MuscleParams& p);

// Make activation_gait_path(label) current for `p`: regenerate it from the
// table gait_table_path(label) and its forces gait_forces_path(label) if it is
// missing or its header doesn't match.
// Writes to a temporary file and renames it into place, so a reader never sees
// a partial file. Returns true if it regenerated. Throws on a missing table or
// I/O errors.
bool ensure_activation_gait(const std::string& label, const MuscleParams& p);
