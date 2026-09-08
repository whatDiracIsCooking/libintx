math(EXPR lmax2 "2*${LIBINTX_MAX_L}")

add_library(
  libintx.md.basis
  EXCLUDE_FROM_ALL
  basis.cc
  )
set_target_properties(libintx.md.basis PROPERTIES OUTPUT_NAME intx.md.basis)

foreach (KET RANGE ${lmax2})

  foreach (BRA RANGE ${lmax2})
    set(target libintx.md4.kernel.${BRA}_${KET})
    add_library(${target} OBJECT EXCLUDE_FROM_ALL md4.kernel.cc)
    target_compile_definitions(
      ${target}
      PRIVATE
      LIBINTX_MD_MD4_KERNEL_BRA=${BRA}
      LIBINTX_MD_MD4_KERNEL_KET=${KET}
      )
    list(APPEND libintx_md_md4_sources $<TARGET_OBJECTS:${target}>)
  endforeach()

  foreach (X RANGE ${LIBINTX_MAX_X})
    set(target libintx.md3.kernel.${X}_${KET})
    add_library(${target} OBJECT EXCLUDE_FROM_ALL md3.kernel.cc)
    target_compile_definitions(
      ${target}
      PRIVATE
      LIBINTX_MD_MD3_KERNEL_BRA=${X}
      LIBINTX_MD_MD3_KERNEL_KET=${KET}
      )
    list(APPEND libintx_md_md3_sources $<TARGET_OBJECTS:${target}>)
  endforeach()

endforeach()

# The two host K engines ride along with the integral library each one drives:
# the integral-direct engine is a driver over libintx::md::IntegralEngine<4>,
# the density-fitted one over IntegralEngine<3>. Neither has sources of its own
# beyond the single translation unit; the algorithms are the headers under
# libintx/gpu/kengine/md/, shared with the device engines.
set(libintx_md_md4_extra_sources kengine.cc)
set(libintx_md_md3_extra_sources df.kengine.cc)

foreach (Centers 2 3 4)
  add_library(
    libintx.md${Centers}
    #EXCLUDE_FROM_ALL
    md${Centers}.cc
    ${libintx_md_md${Centers}_extra_sources}
    ${libintx_md_md${Centers}_sources}
    )
  set_target_properties(libintx.md${Centers} PROPERTIES OUTPUT_NAME intx.md${Centers})
  target_link_libraries(libintx.md${Centers} ${LIBINTX_OPENMP_LIBRARIES})
  install(TARGETS libintx.md${Centers})
endforeach()

target_link_libraries(libintx.md3 libintx.blas)
target_link_libraries(libintx.md3 libintx.md.basis)

target_link_libraries(libintx.md4 libintx.blas)
target_link_libraries(libintx.md4 libintx.md.basis)

install(TARGETS libintx.md.basis)
#install(FILES engine.h DESTINATION libintx/cuda/md)
