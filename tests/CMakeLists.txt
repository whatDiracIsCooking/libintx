add_custom_target(all.tests)

include_directories(${PROJECT_SOURCE_DIR}/include/eigen3)
include_directories(${PROJECT_SOURCE_DIR}/tests)

macro(add_libintx_test name source)
  add_executable(${name} EXCLUDE_FROM_ALL ${source})
  add_test(NAME ${name} COMMAND ${name})
  add_dependencies(all.tests ${name})
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
