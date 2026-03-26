# Native addon

Numbl's Node path can use a native addon for BLAS/LAPACK and FFT acceleration.

## Install behavior

`npm install` now makes a best-effort attempt to build the addon when it can
detect compatible native libraries. A failed native build does not fail the
package install; Numbl falls back to the TypeScript bridge.

Skip the install-time build:

```bash
NUMBL_SKIP_NATIVE_INSTALL=1 npm install
```

`NUMBL_NO_NATIVE=1` also skips the install-time build, but that variable is
primarily for runtime opt-out. An explicit `numbl build-addon` still forces a
build attempt.

Force the install-time build:

```bash
NUMBL_FORCE_NATIVE_BUILD=1 npm install
```

Try the local BLIS/libFLAME/ducc0 fallback during install:

```bash
NUMBL_NATIVE_BUILD_FALLBACK=1 npm install
```

Manual build:

```bash
numbl build-addon
numbl build-addon --with-deps
```

## Library detection

The build uses `pkg-config` when available and searches these package groups by
default:

- BLAS: `blis`, `blis-serial`, `blis-pthread`, `blis-openmp`, `openblas`, `flexiblas`, then `blas`
- LAPACK: `libflame`, `flame`, `lapack`, then `openblas`
- FFT: `fftw3`

Autodetection is capability-based, not package-name-based. Numbl will attempt
an automatic build when it has evidence for:

- BLAS
- LAPACK
- FFT

That evidence can come from `pkg-config` or from explicit `NUMBL_NATIVE_LIBS`.
If only one side of BLAS/LAPACK is discovered through `pkg-config`, the config
script fills in the conventional partner library (`-lblas` or `-llapack`) for
the final link line. When `pkg-config` metadata is unavailable, the install
helper also tries a small set of common system link lines before giving up:

- `-lopenblas -llapack -lfftw3`
- `-lblis -lflame -lfftw3`
- `-lblas -llapack -lfftw3`

OpenBLAS is treated specially only for thread control. Any provider exposing
the Fortran BLAS/LAPACK symbols used by the addon should link. When
`NUMBL_NATIVE_FFT_BACKEND=ducc0` or `NUMBL_NATIVE_DUCC0_INCLUDE_DIRS` is set,
the addon uses DUCC0 instead of FFTW for native FFT entry points.

## Manual overrides

For custom builds or locally compiled libraries, set:

```bash
NUMBL_NATIVE_LIBS="-L/opt/math/lib -lopenblas -lfftw3"
NUMBL_NATIVE_INCLUDE_DIRS="/opt/math/include:/opt/fftw/include"
NUMBL_NATIVE_CFLAGS="-mfma -funroll-loops"
```

On Windows, use `;` instead of `:` in `NUMBL_NATIVE_INCLUDE_DIRS`.

For a local BLIS/libFLAME/ducc0 build:

```bash
NUMBL_NATIVE_FFT_BACKEND=ducc0
NUMBL_NATIVE_DUCC0_INCLUDE_DIRS="/opt/ducc/src"
NUMBL_NATIVE_BLAS_PROVIDER=blis
NUMBL_NATIVE_LAPACK_PROVIDER=libflame
NUMBL_NATIVE_FFT_PROVIDER=ducc0
```

You can also override pkg-config search order:

```bash
NUMBL_NATIVE_BLAS_PKGS="openblas blas"
NUMBL_NATIVE_LAPACK_PKGS="lapack openblas"
NUMBL_NATIVE_FFT_PKGS="fftw3"
```

Enable verbose install diagnostics:

```bash
NUMBL_DEBUG_NATIVE=1 numbl build-addon
```

Disable `-march=native` if needed:

```bash
NUMBL_DISABLE_MARCH_NATIVE=1 numbl build-addon
```

## Build tool resolution

The native build helper tries, in order:

1. npm's configured `node-gyp`
2. a package-local `node-gyp`
3. `npm exec node-gyp`
4. `node-gyp` on `PATH`

This keeps `npm install` and `numbl build-addon` working in both repo checkouts
and npm-installed packages without requiring a globally installed `node-gyp`.

## Local fallback build

Use the helper below to build the preferred fallback stack explicitly:

```bash
npm run build:native-deps
npm run build:addon:deps
```

The helper currently stages BLIS, libFLAME, and DUCC0 under
`.cache/native-deps/` unless you override the source or install roots with the
`NUMBL_*_UPSTREAM_ROOT`, `NUMBL_*_GIT_URL`, `NUMBL_*_GIT_REF`, or
`NUMBL_NATIVE_DEPS_*` environment variables. By default it pins BLIS `2.0`,
libFLAME `5.2.0`, and the `ducc0` branch of the DUCC source tree instead of
building arbitrary upstream heads. The fallback BLIS build defaults to
`NUMBL_NATIVE_DEPS_BLIS_CONFIG=generic` for portability and shorter build
times.
