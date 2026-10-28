# Standard Library

The standard library is the set of MATLAB-surface functions that numbl implements in MATLAB itself rather than as TypeScript `IBuiltin`s. These are thin wrappers, compatibility helpers, and higher-level utilities where writing the logic in MATLAB is more natural than hard-coding it in the interpreter.

The `.m` files live in `src/stdlib/`. Examples include `inputParser`, `readmatrix`, `normalize` and `triangulation`, plus shims under `src/stdlib/shims/` (`+containers`, `+matlab`, `+event`, `uihtml.m`).

## Why it is bundled

The CLI can read `.m` files from disk, but the browser web app cannot. To make the same stdlib available in both, the stdlib files are embedded into the core as a single generated, committed TypeScript module (`src/numbl-core/stdlib-bundle.ts`, written by `scripts/bundle-stdlib.ts`). It exports arrays of `{ name, source }` pairs that `executeCode` injects into every run's `Workspace` (and so its `CodeIndex`) as pre-registered workspace files.

There are two arrays:

- **flat stdlib**: functions and classes that live on the default search path.
- **shims**: compatibility wrappers that need to be resolved through a search-path mechanism (e.g., namespace packages).

## Regenerating the bundle

The generated file is committed and never hand-edited. See [Generated files](../../CONTRIBUTING.md#generated-files) for when to regenerate it.

## Adding a stdlib function

1. Decide between stdlib (`.m`) and builtin (`IBuiltin`). Rough guide: if the behavior is naturally expressible in MATLAB and not performance-critical in the inner loop, use a `.m` file. If it is a primitive or needs tight JIT integration, make it a builtin.
2. Add the `.m` file under `src/stdlib/` with the function or class name matching the file name.
3. Regenerate the bundle (`npm run bundle:stdlib`).
4. Add tests (unit or integration) and verify behavior matches MATLAB.
