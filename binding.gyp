{
  "targets": [
    {
      "target_name": "numbl_addon",
      "sources": [
        "native/numbl_addon.cpp",
        "native/lapack_inv.cpp",
        "native/lapack_qr.cpp",
        "native/lapack_lu.cpp",
        "native/lapack_svd.cpp",
        "native/lapack_matmul.cpp",
        "native/lapack_matmul_complex.cpp",
        "native/lapack_linsolve.cpp",
        "native/lapack_eig.cpp",
        "native/lapack_chol.cpp",
        "native/lapack_qz.cpp",
        "native/lapack_fft.cpp",
        "native/lapack_fft_batch.cpp",
        "native/elemwise.cpp",
        "native/randn.cpp",
        "native/unary_elemwise.cpp",
        "native/lapack_gmres.cpp",
        "native/ops/numbl_ops.c",
        "native/ops/real_binary_elemwise.c",
        "native/ops/complex_binary_elemwise.c",
        "native/ops/real_unary_elemwise.c",
        "native/ops/complex_unary_elemwise.c",
        "native/ops/comparison.c",
        "native/ops/reduce.c",
        "native/ops/bessel.c",
        "native/ops_napi.cpp"
      ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "native/ops",
        "<!@(pkg-config --cflags-only-I fftw3 2>/dev/null | sed 's/-I//g' || true)"
      ],
      "dependencies": [
        "<!(node -p \"require('node-addon-api').gyp\")"
      ],
      "defines": [ "NAPI_DISABLE_CPP_EXCEPTIONS" ],
      "libraries": [
        "-lm",
        "<!@(pkg-config --libs fftw3 2>/dev/null || echo '-lfftw3')",
        "<!@(pkg-config --libs-only-L fftw3 2>/dev/null | sed 's/-L/-Wl,-rpath,/g' || true)"
      ],
      "cflags": [ "-O3", "-march=native", "-fopenmp-simd", "-fno-math-errno", "-ffast-math" ],
      "cflags_c": [ "-O3", "-march=native", "-fopenmp-simd", "-fno-math-errno", "-ffast-math" ],
      "cflags_cc": [ "-std=c++17", "-O3", "-march=native", "-fopenmp-simd", "-fno-math-errno", "-ffast-math" ],
      "conditions": [
        ['OS=="linux"', {
          "cflags_c": [ "-fopenmp" ],
          "cflags_cc": [ "-fopenmp" ],
          "libraries": [ "-lopenblas", "-lmvec", "-fopenmp" ]
        }],
        ['OS=="mac"', {
          # Use Apple's Accelerate framework for BLAS/LAPACK — ships with
          # macOS, so no Homebrew openblas install or keg-only linker path
          # wrangling. The classic Fortran ABI (dgetrf_, etc.) that the
          # native sources declare is provided by Accelerate.
          "libraries": [ "-framework Accelerate" ]
        }]
      ]
    },
    {
      "target_name": "numbl_ops",
      "type": "static_library",
      "sources": [
        "native/ops/numbl_ops.c",
        "native/ops/real_binary_elemwise.c",
        "native/ops/complex_binary_elemwise.c",
        "native/ops/real_unary_elemwise.c",
        "native/ops/complex_unary_elemwise.c",
        "native/ops/comparison.c",
        "native/ops/reduce.c",
        "native/ops/bessel.c"
      ],
      "include_dirs": [ "native/ops" ],
      "cflags_c": [ "-O3", "-march=native", "-fopenmp-simd", "-fno-math-errno", "-ffast-math", "-fPIC" ],
      "conditions": [
        ['OS=="linux"', {
          "cflags_c": [ "-fopenmp" ]
        }]
      ]
    },
    {
      "target_name": "numbl_jit_runtime",
      "type": "static_library",
      "sources": [
        "native/jit_runtime/jit_runtime.c"
      ],
      "include_dirs": [ "native/jit_runtime", "native/ops" ],
      "cflags_c": [ "-O3", "-march=native", "-fopenmp-simd", "-fno-math-errno", "-ffast-math", "-fPIC" ]
    }
  ]
}
