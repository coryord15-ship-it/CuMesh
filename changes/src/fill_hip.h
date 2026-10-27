// fill_hip.h -- host-side declaration of the HIP fill path.
//
// Implemented by fill_hip.hip.cpp (DRAFT -- UNCOMPILED, UNPROVEN
// until built on the target rig; see that file's header). The CPU
// worker (fill_worker.cpp) calls this only when built with
// -DR1_USE_HIP (see BUILD-ON-PC.md); the default build runs the
// host logic in fill_logic.cpp, whose behaviour this path mirrors.
//
// Contract: same inputs and appended-block outputs as
// r1fill::compute_fill, plus diagnostics. Returns 0 on success;
// nonzero on any failure, with `error` set -- the worker turns any
// nonzero into a nonzero exit with no output file, which is the
// seam's exit discipline (spec section 2.1).

#pragma once

#include <cstdint>
#include <string>
#include <vector>

int r1_hip_fill(const float* verts, int64_t num_verts,
                const int64_t* faces, int64_t num_faces,
                float max_hole_perimeter,
                std::vector<float>& appended_verts,
                std::vector<int64_t>& appended_faces,
                std::string& error);
