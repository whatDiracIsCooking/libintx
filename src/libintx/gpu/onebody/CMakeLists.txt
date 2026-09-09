# The device one-electron engine: gpu::md::IntegralEngine<2> and the pieces
# the overlap, kinetic and electron-nuclear-potential kernels share.
#
# The operator kernels themselves go in gpu/overlap/, gpu/kinetic/ and
# gpu/potential_en/ and are added to this target as they land. Start with ONE
# translation unit per operator and only split further if nvcc compile time
# actually demands it: two-centre is (LMAX+1)^2 = 16 kernels at LMAX=3, not the
# (2*LMAX+1)^2 that forces the Coulomb path's one-OBJECT-library-per-pair
# split in gpu/md/CMakeLists.txt.
#
# NB: every .cu added here must also be listed below, or the HIP build silently
# compiles it as plain C++.
if (LIBINTX_HIP)
  set_source_files_properties(
    ../overlap/overlap.cu
    ../potential_en/potential_en.cu
    # ../kinetic/kinetic.cu
    PROPERTIES
    LANGUAGE HIP
    )
endif()

add_library(
  libintx.gpu.md2
  EXCLUDE_FROM_ALL
  basis.cc
  md2.cc
  ../overlap/overlap.cu
  ../potential_en/potential_en.cu
  )
set_target_properties(libintx.gpu.md2 PROPERTIES OUTPUT_NAME intx.gpu.md2)
target_link_libraries(libintx.gpu.md2 libintx.gpu)
target_link_libraries(libintx.gpu.md2 ${libintx_gpu_runtime})

install(TARGETS libintx.gpu.md2)
install(
  FILES
    ${PROJECT_SOURCE_DIR}/src/libintx/gpu/onebody/basis.h
    ${PROJECT_SOURCE_DIR}/src/libintx/gpu/onebody/engine.h
    ${PROJECT_SOURCE_DIR}/src/libintx/gpu/onebody/kernel.h
  DESTINATION libintx/gpu/onebody
  )
install(
  FILES
    ${PROJECT_SOURCE_DIR}/src/libintx/gpu/overlap/overlap.h
  DESTINATION libintx/gpu/overlap
  )
install(
  FILES
    ${PROJECT_SOURCE_DIR}/src/libintx/gpu/potential_en/potential_en.h
  DESTINATION libintx/gpu/potential_en
  )
