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
        "-lopenblas",
        "<!@(pkg-config --libs fftw3 2>/dev/null || echo '-lfftw3')",
        "<!@(pkg-config --libs-only-L fftw3 2>/dev/null | sed 's/-L/-Wl,-rpath,/g' || true)"
      ],
      "cflags_cc": [ "-std=c++17", "-O3", "-march=native" ]
    }
  ]
}
