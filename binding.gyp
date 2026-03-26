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
        "native/elemwise.cpp"
      ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "<!@(node scripts/native-addon-config.mjs include-dirs)"
      ],
      "dependencies": [
        "<!(node -p \"require('node-addon-api').gyp\")"
      ],
      "defines": [
        "NAPI_DISABLE_CPP_EXCEPTIONS",
        "<!@(node scripts/native-addon-config.mjs defines)"
      ],
      "libraries": [
        "<!@(node scripts/native-addon-config.mjs libraries)"
      ],
      "cflags_cc": [
        "-std=c++17",
        "-O3",
        "<!@(node scripts/native-addon-config.mjs cflags-cc)"
      ]
    }
  ]
}
