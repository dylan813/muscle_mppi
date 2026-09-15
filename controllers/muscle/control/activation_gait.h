#pragma once

// Activation gait generation for the muscle variant.
//
// An activation gait is the muscle-space version of a joint-space source gait:
// for every phase column of the source (joint positions + velocities, from
// controllers/pd/gaits/), each joint's activation pair is the point on its
// torque-balance line at MuscleParams::stiffness that produces the joint's bias
// torque (gravity + Coriolis, suspended model, no contact) — hill_invert_torque().
//
// Generated files live in controllers/muscle/gaits/ (gitignored) and start with
// one header line recording what they were generated from:
//   # activation_gait source=<key> stiffness=<value> muscle=<fingerprint>
// The gait loader skips it (non-numeric), as does np.loadtxt ('#').
// ensure_activation_gait() regenerates a file whenever that line doesn't match
// the current source key and muscle parameters (including stiffness).

#include <string>

#include "../utils/tasks.h"

// Repo-relative path of a joint-space source gait, e.g. "FAST_0_1_10cm" ->
// controllers/pd/gaits/FAST/gait_FAST_0_1_10cm.tsv.
std::string source_gait_path(const std::string& key);

// Repo-relative path of the generated activation gait for a source key, e.g.
// "FAST_0_1_10cm" -> controllers/muscle/gaits/FAST/activation_gait_FAST_0_1_10cm.tsv.
std::string activation_gait_path(const std::string& key);

// Hash of every muscle parameter the generated gait depends on (stiffness,
// lce_min/max, phi_min/max, pFLmax, FVmax, vmax, peak_force) plus a generator
// version, as 16 hex digits.
std::string muscle_fingerprint(const MuscleParams& p);

// The header line (without "# " and newline) a gait generated from `key` with `p` carries.
std::string activation_gait_header(const std::string& key, const MuscleParams& p);

// Make activation_gait_path(key) current for `p`: regenerate it from
// source_gait_path(key) if it is missing or its header doesn't match. Writes to
// a temporary file and renames it into place, so a reader never sees a partial
// file. Returns true if it regenerated. Throws on missing sources or I/O errors.
bool ensure_activation_gait(const std::string& key, const MuscleParams& p);
