#!/usr/bin/env bash
# Per-project settings for everything in devtools/. EDIT THIS FILE, not the
# scripts -- the scripts are meant to survive being copied between repos
# unchanged, and this is the one place that knows what the project is called
# and how its tests run.
#
# Sourced (never executed) by devcontainer.sh, worktree.sh, doctor.sh,
# build.sh, cpp-tier.sh and cpp-smoke.sh. Every value uses ${VAR:-default} so
# an environment variable still wins for a one-off:
#
#   BUILD_JOBS=4 devtools/cpp-tier.sh
#
# The one thing this file CANNOT reach is .devcontainer/devcontainer.json --
# JSON cannot source shell. The volume names, the container name and the .git
# bind path are spelled out there too; PROJECT_NAME below must match, or
# `worktree.sh rm` and `worktree.sh gc` will not recognise this project's
# volumes as its own. doctor.sh checks that the two agree.
#
# The C++ side has a second config file with the same job: CMakePresets.json
# holds the build settings (build type, CUDA architecture, LIBINTX_MAX_L).
# What lives HERE is only which preset the scripts drive -- see CMAKE_PRESET /
# CTEST_PRESET below.
#
# NB libintx is a C++ project with no Python test suite of its own, so the
# pytest-shaped scripts of the upstream devtools set (prepush-tests.sh,
# slow-tier.sh) are deliberately absent here rather than carried along empty.
# The gate is the C++ one: devtools/cpp-smoke.sh on push, devtools/cpp-tier.sh
# for the real thing.

# --- identity -------------------------------------------------------------

# Prefix for this project's docker volumes and per-worktree build images, and
# the name of the devcontainer. Must match devcontainer.json (see above).
# It ends up in docker resource names, so keep it lowercase.
PROJECT_NAME=${PROJECT_NAME:-libintx}

# Which devcontainer.json devcontainer.sh drives. Relative paths resolve
# against the repo root, so this works from any cwd and from any worktree.
#
#   .devcontainer/cuda/devcontainer.json   the CUDA image: nvcc, cmake, ninja
#                                          and a device. Needs an NVIDIA GPU
#                                          and the container toolkit.
#   .devcontainer/devcontainer.json        the light CPU image.
#
# Points at the CUDA variant because that is where the GPU half of libintx
# actually builds; the host half (boys, md2/3/4, the host K engine) builds
# anywhere with a C++20 compiler, LAPACK and CBLAS.
DEVCONTAINER_CONFIG=${DEVCONTAINER_CONFIG:-.devcontainer/cuda/devcontainer.json}

# --- C++ build (CMake) ------------------------------------------------------

# Which CMakePresets.json presets devtools/build.sh and cpp-tier.sh drive. The
# presets themselves -- build type, CUDA architecture, LIBINTX_MAX_L -- live in
# CMakePresets.json; only the CHOICE lives here.
#
#   default      Release + CUDA, CMAKE_CUDA_ARCHITECTURES=native, build/
#   workstation  Release, CPU only, build-workstation/ -- the bare-host preset
#   debug / asan
#   ampere / hopper / portable / hip   pin the architecture or the API
#
# Override for one run rather than editing:
#   CMAKE_PRESET=workstation CTEST_PRESET=workstation devtools/cpp-tier.sh
#
# CTEST_PRESET is separate because not every configure preset has a matching
# test preset -- only default, workstation, debug and asan do. Set it empty to
# configure and build without running ctest.
CMAKE_PRESET=${CMAKE_PRESET:-default}
CTEST_PRESET=${CTEST_PRESET:-default}

# Parallel compile jobs for `cmake --build`. libintx explodes its MD kernels
# into one translation unit per (bra,ket) angular-momentum pair -- (2*LMAX+1)^2
# of them for md4 alone, each instantiated at every L -- so this is a wide,
# memory-hungry build. Empty lets Ninja pick (cores + 2), which is what
# OOM-kills a small box; an OOM-killed compiler surfaces as a bare
# `ninja: build stopped` with nothing about memory in it.
BUILD_JOBS=${BUILD_JOBS:-8}

# --- tests ----------------------------------------------------------------

# How the suite is invoked. ctest, not pytest: libintx's tests are doctest
# executables registered with add_test(), driven through the ctest preset.
TEST_CMD=${TEST_CMD:-ctest}

# The test targets a build has to produce before ctest has anything to run.
# libintx declares every test executable EXCLUDE_FROM_ALL and collects them
# under the `all.tests` custom target, so a plain `cmake --build` produces the
# libraries and NO tests -- and ctest then reports "No tests were found",
# which reads like a configuration problem rather than a missing target. The
# build presets in CMakePresets.json name both targets for this reason.
BUILD_TARGETS=${BUILD_TARGETS:-"all all.tests"}

# What the push gate (devtools/cpp-smoke.sh) runs, as a ctest -R regex. Keep
# it to the cheap, host-only cases: the gate should cost seconds, and anything
# needing a device or a long reference sum belongs in cpp-tier.sh.
#
# Both K engine tests are in here because their references are independent
# brute-force sums -- they are the cases most likely to catch a real regression
# in the digest and in the DF assembly, and at LMAX<=2 they cost a few seconds.
SMOKE_TESTS=${SMOKE_TESTS:-'^(pure\.test|boys\.test|libintx\.kengine\.test|libintx\.df\.kengine\.test)$'}

# Tests the smoke gate runs only when a device is present. Empty disables the
# GPU half of the gate.
SMOKE_GPU_TESTS=${SMOKE_GPU_TESTS:-'^(boys\.gpu\.test|libintx\.gpu\.kengine\.test)$'}

# The preset the smoke gate builds and tests with. `workstation` rather than
# `default` so a push from a machine with no GPU is still gated rather than
# skipped -- `default` cannot even configure without a device, because
# CMAKE_CUDA_ARCHITECTURES=native queries one.
SMOKE_PRESET=${SMOKE_PRESET:-workstation}

# Host CPU bounds for a devcontainer, applied by devcontainer.sh on `up` and
# `rebuild`. Both empty = unbounded, and nothing is applied.
#
#   CPUSET  pin the container to specific host cores, e.g. 0-11
#   CPUS    cap it at n cores' worth of CPU quota, e.g. 8
#
# These bound EVERYTHING the container runs, where BUILD_JOBS only bounds the
# compiler. A container is per workspace folder, so these are per worktree:
# two containers pinned to disjoint cores are the only way concurrent
# worktrees stop stealing each other's timings.
CPUSET=${CPUSET:-}
CPUS=${CPUS:-}

# Where cpp-tier.sh and build.sh write their run logs. Relative paths resolve
# against the repo root. Gitignored.
SLOW_TIER_REPORTS=${SLOW_TIER_REPORTS:-.slow-tier-reports}

# --- environment ----------------------------------------------------------

# gh reads GH_TOKEN (then GITHUB_TOKEN) and nothing else, but a PAT scoped to
# this one repository should not be exported globally on the host. So the host
# keeps it under a project-specific name and the mapping happens here.
if [ -n "${GH_TOKEN_LIBINTX:-}" ]; then
  export GH_TOKEN=$GH_TOKEN_LIBINTX
fi

# Where to look for a project virtualenv, in order. libintx has no Python
# dependency for its C++ build; this only feeds `find_tool`, which the scripts
# use so a tool installed in a venv is found without activating it (cmake-lint
# and the python/ bindings' tooling are the cases).
VENV_PATHS=${VENV_PATHS:-.venv /opt/venv}

# Optional tools doctor.sh reports on, one `name:what is lost without it` per
# line. A missing one is a WARN (a slice of the suite will skip), never a FAIL.
#
# Most of this list is absent on a bare host and present in the CUDA container,
# which is the point: run doctor.sh on both sides and the warnings tell you
# which half of the workflow you are on.
DOCTOR_OPTIONAL_TOOLS=${DOCTOR_OPTIONAL_TOOLS:-"docker:devcontainer, compose services and worktree containers
gh:the PR flow (also needs GH_TOKEN)
cmake:configuring and building the tree at all
ninja:the generator every preset uses
nvcc:the CUDA half -- libintx.gpu, gpu.md3/md4, the J engine and the device K engine
hipcc:the HIP half (-DLIBINTX_HIP=ON)
nsys:Nsight Systems profiling from inside the container
ccache:warm rebuilds
clang-format:the C++ formatting pass, run by hand
cmake-format:the CMake formatting pass, run by hand"}

# Paths that must exist for some slice of the suite to build or run, one
# `path:what breaks without it` per line.
#
# libintx vendors its heavy dependencies in-tree rather than fetching them --
# eigen3, cutlass/cute, doctest and pybind11 are all checked in -- so the
# entries here are the ones a shallow or filtered clone can lose, plus the two
# system headers libintx.blas needs and CMake does not check for.
DOCTOR_REQUIRED_PATHS=${DOCTOR_REQUIRED_PATHS:-"include/eigen3/Eigen/Core:Eigen, which the tests and libintx/math include
tests/doctest.h:the test framework -- every tests/*.test.cc includes it
python/pybind11/CMakeLists.txt:the Python bindings (-DLIBINTX_PYTHON=ON)"}
