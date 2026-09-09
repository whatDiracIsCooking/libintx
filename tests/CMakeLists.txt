add_custom_target(all.tests)

include_directories(${PROJECT_SOURCE_DIR}/include/eigen3)
include_directories(${PROJECT_SOURCE_DIR}/tests)

# Tiering. Every registered test carries exactly one label, so `ctest -L fast`
# and `ctest -L slow` partition the CPU suite and `-L gpu` is the device half:
#
#   fast -- seconds to a couple of minutes. The per-change loop.
#   slow -- the (bra,ket) angular-momentum sweeps. Minutes at LIBINTX_MAX_L=2,
#           hours at 3.
#   gpu  -- needs a device. Only registered when LIBINTX_CUDA/LIBINTX_HIP is on.
#
# The label is derived from the test's name so that a new add_libintx_test()
# call lands in the right tier without a second edit: anything with `.gpu.` in
# its name is gpu, anything named in LIBINTX_SLOW_TESTS is slow, the rest fast.
set(LIBINTX_SLOW_TESTS libintx.md2.test libintx.md3.test libintx.md4.test)

# Timeouts. ctest's default is 1500 s, which is *shorter than a legitimate run*
# of libintx.md4.test at the default LIBINTX_MAX_L=3 -- 1 h 56 min, which ctest
# reported as `***Timeout` and so hid a genuine assertion failure for as long
# as it existed (issue #15). These are backstops for a wedged machine, not
# performance budgets, so they are deliberately loose.
#
# Cost climbs steeply with LIBINTX_MAX_L: the sweeps walk every (bra,ket) pair
# and the reference oracle behind them is itself ~8x per unit of L. So scale
# with the MAX_L the tree was configured with rather than hardcode one number
# that is either useless at 2 or wrong at 3.
set(LIBINTX_TEST_TIMEOUT_FAST 300)
set(LIBINTX_TEST_TIMEOUT_SLOW 1200)
set(_l 2)
while (_l LESS ${LIBINTX_MAX_L})
  math(EXPR LIBINTX_TEST_TIMEOUT_FAST "${LIBINTX_TEST_TIMEOUT_FAST}*8")
  math(EXPR LIBINTX_TEST_TIMEOUT_SLOW "${LIBINTX_TEST_TIMEOUT_SLOW}*8")
  math(EXPR _l "${_l}+1")
endwhile()
unset(_l)
message(
  STATUS
  "libintx tests: LIBINTX_MAX_L=${LIBINTX_MAX_L}, "
  "timeouts fast=${LIBINTX_TEST_TIMEOUT_FAST}s slow=${LIBINTX_TEST_TIMEOUT_SLOW}s"
  )

macro(add_libintx_test name source)
  add_executable(${name} EXCLUDE_FROM_ALL ${source})
  add_test(NAME ${name} COMMAND ${name})
  add_dependencies(all.tests ${name})
  if ("${name}" MATCHES "\\.gpu\\.")
    set(_label gpu)
    set(_timeout ${LIBINTX_TEST_TIMEOUT_SLOW})
  elseif ("${name}" IN_LIST LIBINTX_SLOW_TESTS)
    set(_label slow)
    set(_timeout ${LIBINTX_TEST_TIMEOUT_SLOW})
  else()
    set(_label fast)
    set(_timeout ${LIBINTX_TEST_TIMEOUT_FAST})
  endif()
  set_tests_properties(${name} PROPERTIES LABELS ${_label} TIMEOUT ${_timeout})
  unset(_label)
  unset(_timeout)
endmacro()

add_libintx_test(pure.test pure.test.cc)

add_libintx_test(boys.test boys.test.cc)
target_link_libraries(boys.test libboys)

add_libintx_test(libintx.md.test libintx.md.test.cc)
target_link_libraries(libintx.md.test libboys)

option(LIBINTX_LIBINT2 "" OFF)
if (LIBINTX_LIBINT2)
  find_package(Libint2 REQUIRED)
  add_library(libintx.test.reference libintx.reference.cc)
  set_target_properties(libintx.test.reference PROPERTIES OUTPUT_NAME intx.test.reference)
  target_compile_definitions(libintx.test.reference PUBLIC LIBINTX_TEST_REFERENCE)
  target_link_libraries(libintx.test.reference PRIVATE Libint2::cxx)
endif()

if (LIBINTX_CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++20")
  enable_language(CUDA)
  add_libintx_test(boys.gpu.test boys.gpu.test.cu)
  target_link_libraries(boys.gpu.test libboys libintx.gpu)
endif()

add_libintx_test(libintx.kengine.test libintx.kengine.test.cc)
target_link_libraries(libintx.kengine.test libintx.md4 libboys)

# The density-fitted host K engine lives in libintx.md3; its exact-fit case
# runs the direct engine alongside it, so this needs both libraries.
add_libintx_test(libintx.df.kengine.test libintx.df.kengine.test.cc)
target_link_libraries(libintx.df.kengine.test libintx.md3 libintx.md4 libboys)

# The conventional (integral-direct) host J engine. Also covers the fused
# J+K sweep, which is why it links the K engine's library too -- same target.
add_libintx_test(libintx.jengine.test libintx.jengine.test.cc)
target_link_libraries(libintx.jengine.test libintx.md4 libboys)

foreach (n 2 3 4)
  # tests
  add_libintx_test(libintx.md${n}.test libintx.md${n}.test.cc)
  target_link_libraries(libintx.md${n}.test libintx.md${n})
  # perf benchmarks
  add_executable(libintx.md${n}.perf EXCLUDE_FROM_ALL libintx.md${n}.perf.cc)
  target_link_libraries(libintx.md${n}.perf libintx.md${n})
  if (TARGET libintx.test.reference)
    target_link_libraries(libintx.md${n}.perf libintx.test.reference)
  endif()
endforeach()

foreach (n 3 4)
  if (NOT TARGET libintx.gpu.md${n})
    continue()
  endif()
  # tests
  add_libintx_test(libintx.gpu.md${n}.test libintx.gpu.md${n}.test.cc)
  target_link_libraries(libintx.gpu.md${n}.test libintx.gpu.md${n})
  # perf benchmarks
  add_executable(libintx.gpu.md${n}.perf EXCLUDE_FROM_ALL libintx.gpu.md${n}.perf.cc)
  target_link_libraries(libintx.gpu.md${n}.perf libintx.gpu.md${n})
  if (TARGET libintx.test.reference)
    target_link_libraries(libintx.gpu.md${n}.perf libintx.test.reference)
  endif()
endforeach()

if (TARGET libintx.gpu.md2)
  # The device McMurchie-Davidson E coefficients against the host ones. This is
  # the piece all three one-electron device kernels are built on, so it is
  # checked on its own rather than only through an operator.
  if (LIBINTX_HIP)
    set_source_files_properties(
      libintx.gpu.e2.test.cu
      PROPERTIES
      LANGUAGE HIP
      )
  endif()
  add_libintx_test(libintx.gpu.e2.test libintx.gpu.e2.test.cu)
  target_link_libraries(libintx.gpu.e2.test libintx.gpu)
  # The harness the three operator issues add cases to; a direct mirror of
  # libintx.md2.test, which is why it links the host engine too.
  add_libintx_test(libintx.gpu.md2.test libintx.gpu.md2.test.cc)
  target_link_libraries(libintx.gpu.md2.test libintx.gpu.md2 libintx.md2 libboys)
endif()

if (TARGET libintx.gpu.md4)
  # The device K engine is checked against the host one, so this pulls in both.
  add_libintx_test(libintx.gpu.kengine.test libintx.gpu.kengine.test.cc)
  # The DF device engine is in the same test, checked against the host DF one,
  # which is what pulls in the two 3-centre libraries.
  target_link_libraries(
    libintx.gpu.kengine.test
    libintx.gpu.md4 libintx.gpu.md3 libintx.md4 libintx.md3 libboys
    )
  # Likewise the conventional device J engine. This is NOT the density-fitted
  # engine below -- different algorithm, different target.
  add_libintx_test(libintx.gpu.jengine.direct.test libintx.gpu.jengine.direct.test.cc)
  target_link_libraries(
    libintx.gpu.jengine.direct.test libintx.gpu.md4 libintx.md4 libboys
    )
  # Derivative ERI batches, three- and four-centre, against finite differences
  # of the engines' own value path. Both device libraries because the
  # three-centre derivative is in libintx.gpu.md3 and the four-centre one in
  # libintx.gpu.md4; both host libraries because the interface case checks that
  # the host overrides throw.
  add_libintx_test(libintx.gpu.deriv.test libintx.gpu.deriv.test.cc)
  target_link_libraries(
    libintx.gpu.deriv.test
    libintx.gpu.md4 libintx.gpu.md3 libintx.md4 libintx.md3 libboys
    )
endif()

if (TARGET libintx.gpu.eri)
  # The full-ERI formats are checked against libintx::md::reference directly
  # rather than against an engine, so this needs the reference headers and the
  # Boys function but neither host engine.
  add_libintx_test(libintx.gpu.eri.jformat.test libintx.gpu.eri.jformat.test.cc)
  target_link_libraries(libintx.gpu.eri.jformat.test libintx.gpu.eri libboys)
  # The K format is additionally checked against the host K engine, which is
  # where libintx.md4 comes in.
  add_libintx_test(libintx.gpu.eri.kformat.test libintx.gpu.eri.kformat.test.cc)
  target_link_libraries(
    libintx.gpu.eri.kformat.test libintx.gpu.eri libintx.md4 libboys
    )
endif()

if (TARGET libintx.gpu.jengine)
  add_libintx_test(libintx.gpu.jengine.test libintx.gpu.jengine.test.cc)
  target_link_libraries(libintx.gpu.jengine.test libintx.gpu.jengine)
endif()
