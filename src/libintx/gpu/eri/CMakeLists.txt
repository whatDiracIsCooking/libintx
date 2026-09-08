if (LIBINTX_HIP)
  set_source_files_properties(
    eri.jformat.cu
    PROPERTIES
    LANGUAGE HIP
    )
endif()

# One library for both formats: they share eri/format.h, and a caller that
# wants J as a GEMV usually wants K the same way.
add_library(
  libintx.gpu.eri
  EXCLUDE_FROM_ALL
  eri.cc
  eri.jformat.cu
  )
set_target_properties(libintx.gpu.eri PROPERTIES OUTPUT_NAME intx.gpu.eri)
target_link_libraries(libintx.gpu.eri libintx.gpu)
# The integrals themselves come from the four-centre device engine.
target_link_libraries(libintx.gpu.eri libintx.gpu.md4)

install(TARGETS libintx.gpu.eri)
install(
  FILES ${PROJECT_SOURCE_DIR}/src/libintx/gpu/eri.h
  DESTINATION libintx/gpu
  )
