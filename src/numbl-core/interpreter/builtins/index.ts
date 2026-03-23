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

export { getIBuiltin, buildIBuiltinHelpers } from "./types.js";
export type { IBuiltin } from "./types.js";
