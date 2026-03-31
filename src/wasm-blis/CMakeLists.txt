cmake_minimum_required(VERSION 3.16)
project(wasm_blis LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)

include(FetchContent)

FetchContent_Declare(xsimd
  GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
  GIT_TAG        14.1.0
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(xsimd)

FetchContent_Declare(eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(eigen)

# Pre-built OpenBLAS for WASM (single-threaded, built via emmake make)
set(OPENBLAS_ROOT "/tmp/openblas-wasm" CACHE PATH "OpenBLAS built for WASM")
set(WASM_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/../../public/wasm-kernels")

if(EMSCRIPTEN)
  set(WASM_COMPILE_FLAGS
    -O3 -flto -msimd128 -mrelaxed-simd
    -fno-exceptions -fno-rtti
    -fno-math-errno -ffp-contract=fast
    -fno-signed-zeros -fno-trapping-math -fassociative-math
    -DNDEBUG
  )
  set(WASM_LINK_FLAGS
    -O3 -flto -msimd128 -mrelaxed-simd --no-entry
    "SHELL:-s STANDALONE_WASM=1"
    "SHELL:-s INITIAL_MEMORY=67108864"
    "SHELL:-s ALLOW_MEMORY_GROWTH=1"
    "SHELL:-s TOTAL_STACK=8388608"
  )

  # Helper: create a standalone .wasm target
  function(add_wasm_kernel name sources exports)
    add_executable(${name} ${sources})
    target_compile_options(${name} PRIVATE ${WASM_COMPILE_FLAGS})
    target_link_options(${name} PRIVATE ${WASM_LINK_FLAGS}
      "SHELL:-s EXPORTED_FUNCTIONS=['_malloc','_free','${exports}']"
    )
    set_target_properties(${name} PROPERTIES SUFFIX ".wasm")
    add_custom_command(TARGET ${name} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory "${WASM_OUTPUT_DIR}"
      COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:${name}>" "${WASM_OUTPUT_DIR}/${name}.wasm"
      COMMENT "Copied ${name}.wasm"
    )
  endfunction()

  # 1. Our BLIS xsimd kernel
  add_wasm_kernel(blis-dgemm dgemm.cpp _numbl_dgemm_f64)
  target_link_libraries(blis-dgemm PRIVATE xsimd)

  # 2. Eigen kernel
  add_wasm_kernel(eigen-dgemm eigen_dgemm.cpp _numbl_eigen_dgemm_f64)
  target_link_libraries(eigen-dgemm PRIVATE Eigen3::Eigen)

  # 3. OpenBLAS kernel
  add_wasm_kernel(openblas-dgemm openblas_dgemm.cpp _numbl_openblas_dgemm_f64)
  target_include_directories(openblas-dgemm PRIVATE "${OPENBLAS_ROOT}")
  target_link_libraries(openblas-dgemm PRIVATE "${OPENBLAS_ROOT}/libopenblas_wasm128-r0.3.32.a")
endif()
