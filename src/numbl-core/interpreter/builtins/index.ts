/**
 * Interpreter builtins: register all and re-export.
 *
 * Import order matters — each module registers its builtins on import.
 */

import "./math.js";
import "./arithmetic.js";
import "./complex.js";
import "./predicates.js";
import "./logical.js";
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
import "./datetime.js";
import "./sparse.js";
import "./sparse-solvers.js";
import "./graph.js";
import "./gallery.js";
import "./special-math.js";
import "./geometry.js";
import "./misc.js";
import "./dictionary.js";
import "./hash.js";
import "./help-text.js";
import "./output.js";
import "./higher-order.js";
import "./file-io.js";
import "./filesystem.js";
import "./web.js";
import "./path-env.js";
import "./ode.js";
import "./gmres.js";
import "./eigs.js";

import "./scope.js";
// The plot builtins register at module load.
import "../../plotting/builtins.js";

export {
  getIBuiltin,
  getIBuiltinHelp,
  getIBuiltinNargin,
  getAllIBuiltinNames,
  isBuiltin,
} from "../../runtime/builtinRegistry.js";
export type { IBuiltin, BuiltinHelp } from "../../runtime/builtinRegistry.js";
export { builtin, makeBuiltin } from "./builtin.js";
export type {
  BuiltinSpec,
  BuiltinContext,
  ScopedBuiltinContext,
  Arity,
} from "./builtin.js";
export { resetAppdataStore } from "./misc.js";
