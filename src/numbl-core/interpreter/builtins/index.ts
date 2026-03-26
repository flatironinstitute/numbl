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
import "./sparse.js";
import "./special-math.js";
import "./misc.js";

export {
  getIBuiltin,
  getIBuiltinNargin,
  getAllIBuiltinNames,
  buildIBuiltinHelpers,
  setDynamicRegisterHook,
  inferJitType,
} from "./types.js";
export type { IBuiltin, IBuiltinResolution } from "./types.js";

// Register IBuiltin + special builtin names so isBuiltin() recognizes them
import { registerExtraBuiltinNames } from "../../helpers/registry.js";
import { getAllIBuiltinNames as _getAllIBuiltinNames } from "./types.js";
import { SPECIAL_BUILTIN_NAMES } from "../../runtime/specialBuiltins.js";
registerExtraBuiltinNames(_getAllIBuiltinNames());
registerExtraBuiltinNames(SPECIAL_BUILTIN_NAMES);
