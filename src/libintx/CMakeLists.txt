add_library(libintx.blas blas.cc)

if (LIBINTX_CBLAS_H)
  target_compile_definitions(libintx.blas PRIVATE LIBINTX_CBLAS_H=${LIBINTX_CBLAS_H})
endif()

if (LIBINTX_LAPACKE_H)
  target_compile_definitions(libintx.blas PRIVATE LIBINTX_LAPACKE_H=${LIBINTX_LAPACKE_H})
endif()

if (LIBINTX_INTEL_MKL)
  target_compile_definitions(libintx.blas PRIVATE LIBINTX_INTEL_MKL)
  target_link_libraries(libintx.blas mkl_rt)
endif()

if (LIBINTX_APPLE_ACCELERATE)
  target_compile_definitions(libintx.blas PRIVATE LIBINTX_APPLE_ACCELERATE)
endif()

if (TARGET LAPACK::LAPACK)
  target_link_libraries(libintx.blas LAPACK::LAPACK)
endif()

# blas.cc calls LAPACKE_dsygvd, but find_package(LAPACK) locates the Fortran
# LAPACK -- on a platform where the two are packaged separately (a plain
# OpenBLAS on Linux, where lapacke is its own -dev package) the tree configures
# cleanly, compiles, and then fails at LINK with an undefined LAPACKE_dsygvd.
# Where LAPACK::LAPACK already provides the C interface -- Apple Accelerate,
# MKL -- this finds nothing and changes nothing.
if (NOT LIBINTX_APPLE_ACCELERATE AND NOT LIBINTX_INTEL_MKL)
  find_library(LIBINTX_LAPACKE_LIBRARY NAMES lapacke)
  mark_as_advanced(LIBINTX_LAPACKE_LIBRARY)
  if (LIBINTX_LAPACKE_LIBRARY)
    message(STATUS "Found LAPACKE: ${LIBINTX_LAPACKE_LIBRARY}")
    target_link_libraries(libintx.blas ${LIBINTX_LAPACKE_LIBRARY})
  endif()
endif()

install(
  FILES forward.h shell.h orbital.h array.h math.h engine.h utility.h
        jengine.h kengine.h
  DESTINATION libintx
  )
