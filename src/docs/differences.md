# Differences from MATLAB

numbl aims for MATLAB compatibility but currently has some behavioral differences. The list below reflects the present state of numbl — many of these gaps may be closed in future releases as the project matures.

## Performance

Browser mode is designed for development, teaching, and sharing. The CLI with the native addon is significantly faster but still generally slower than MATLAB. Performance is actively improving.

## Numeric Types

All numeric values are currently double-precision. Single-precision (`single`) and integer types (`int8`, `uint16`, etc.) are not yet supported. Integer type names are recognized only in `fread`/`fwrite` format strings. Support for additional numeric types may be added later.

## Path Management

numbl has no notion of a saved path, so `savepath` is a no-op. `addpath` does work, but only for the duration of the current session. To make directories available persistently, use [mip](https://mip.sh) to install reusable libraries, or extend the search path via the CLI `--path` flag or the `NUMBL_PATH` environment variable.

## Caught Exceptions

In a `try ... catch ME` block, `ME` is a plain struct with `message`, `identifier`, and `stack` fields rather than a full `MException` object. Field access works the same way in both environments, but `isfield(ME, 'message')` returns `true` in numbl and `false` in MATLAB (since MATLAB's `MException` is an opaque object). A future version of numbl may introduce a proper `MException` class.

## Struct Arrays

Struct arrays in numbl are stored as flat (1D) sequences rather than full N-D shapes. `numel`, field access, indexed assignment, and iteration behave the same as in MATLAB, but operations that depend on a specific 2D layout (such as `reshape` or shape-sensitive concatenation) may produce a different result. Full multidimensional struct array shapes may be supported in a future release.

## Stubbed Commands

A handful of MATLAB commands are accepted by numbl but currently do nothing meaningful. They exist so that ported code keeps running without modification:

- `clear`, `clc`, `clf` — silently no-op
- `lastwarn` — returns `''`
- `pathdef` — returns `''`
- `listfonts` — returns `{}`

These may gain real implementations as the relevant subsystems land.

## Currently Unsupported Features

The following MATLAB features are not implemented today. Some may be added in future releases; others are unlikely to ever be supported.

- Single-precision and integer numeric types
- Parallel computing (`parfor`, `spmd`)
- GPU arrays
- MEX interface (numbl provides its own equivalent: `.js` user functions can bind to compiled `.wasm` modules or native shared libraries via `// wasm:` and `// native:` directives)
- Simulink and toolboxes
- Java / .NET integration
- App Designer / GUI
- Metaclasses and advanced class introspection
