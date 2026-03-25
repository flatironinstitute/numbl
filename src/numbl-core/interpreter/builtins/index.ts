/**
 * Interpreter builtins: register all and re-export.
 *
 * Import order matters — each module registers its builtins on import.
 */

import "./math.js";
import "./arithmetic.js";
import "./complex.js";
import "./predicates.js";
import "./utility.js";
import "./introspection.js";
import "./array-construction.js";
import "./reductions.js";
import "./strings.js";
import "./set-operations.js";
import "./linear-algebra.js";
import "./fft.js";

export { getIBuiltin, buildIBuiltinHelpers, inferJitType } from "./types.js";
export type { IBuiltin, IBuiltinResolution } from "./types.js";
