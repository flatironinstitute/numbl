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
import "./array-manipulation.js";
import "./array-extras.js";
import "./validation.js";
import "./numerical.js";
import "./type-constructors.js";
import "./string-extras.js";
import "./prng.js";
import "./cell-struct.js";
import "./time-system.js";

export {
  getIBuiltin,
  getAllIBuiltinNames,
  buildIBuiltinHelpers,
  inferJitType,
} from "./types.js";
export type { IBuiltin, IBuiltinResolution } from "./types.js";
