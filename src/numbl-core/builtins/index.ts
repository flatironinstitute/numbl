/**
 * Builtin functions - main entry point
 *
 * This file exports all builtin function registration and utilities.
 */

// Export registry functions
export {
  getBuiltin,
  isBuiltin,
  getAllBuiltinNames,
  getBuiltinNargin,
  registerExtraBuiltinNames,
} from "./registry.js";

// Export constants
export { getConstant, getAllConstantNames } from "./constants.js";

export { getDummyBuiltinNames } from "./dummy.js";

// Registration functions for builtins still served via rt.builtins fallback
// (sparse-matrix support and other functions not yet covered by IBuiltins).
import { registerMathFunctions } from "./math.js";
import { registerReductionFunctions } from "./reduction/register-reduction-functions.js";
import { registerLinearAlgebraFunctions } from "./linear-algebra/register-linear-algebra-functions.js";
registerMathFunctions();
registerReductionFunctions();
registerLinearAlgebraFunctions();
