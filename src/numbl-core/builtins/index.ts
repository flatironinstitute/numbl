/**
 * Builtin functions - main entry point
 *
 * This file exports all builtin function registration and utilities.
 */

// Export registry types and functions
export type { BuiltinFn, BuiltinFnBranch } from "./registry.js";
export {
  builtinSingle,
  getBuiltin,
  findBuiltinBranch,
  isBuiltin,
  getAllBuiltinNames,
  getBuiltinNargin,
  registerExtraBuiltinNames,
} from "./registry.js";

// Export constants
export {
  getConstant,
  getConstantType,
  getAllConstantNames,
} from "./constants.js";

export { getDummyBuiltinNames } from "./dummy.js";

// Import registration functions still needed for legacy fallback.
// These provide sparse-matrix support, assert, graphics stubs, and
// other functionality not yet covered by interpreter IBuiltins.
import { registerMathFunctions } from "./math.js";
import { registerReductionFunctions } from "./reduction/register-reduction-functions.js";
import { registerLinearAlgebraFunctions } from "./linear-algebra/register-linear-algebra-functions.js";
import { registerMiscFunctions } from "./misc.js";
import { registerGraphicsFunctions } from "./graphics.js";
import { registerDummyFunctions } from "./dummy.js";

registerMathFunctions();
registerReductionFunctions();
registerLinearAlgebraFunctions();
registerMiscFunctions();
registerGraphicsFunctions();
registerDummyFunctions();
