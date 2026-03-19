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
} from "./registry.js";

// Export constants
export {
  getConstant,
  getConstantType,
  getAllConstantNames,
} from "./constants.js";

// Import all registration functions
import { registerMathFunctions } from "./math.js";
import { registerPrngFunctions } from "./prng.js";
import { registerArrayFunctions } from "./array.js";
import { registerArrayManipulationFunctions } from "./array-manipulation.js";
import { registerIntrospectionFunctions } from "./introspection.js";
import { registerReductionFunctions } from "./reduction/register-reduction-functions.js";
import { registerStringFunctions } from "./string.js";
import { registerLinearAlgebraFunctions } from "./linear-algebra/register-linear-algebra-functions.js";
import { registerMiscFunctions } from "./misc.js";
import { registerGraphicsFunctions } from "./graphics.js";
import { registerValidatorFunctions } from "./validators.js";
import { registerDummyFunctions } from "./dummy.js";
export { getDummyBuiltinNames } from "./dummy.js";
import { registerNumericalFunctions } from "./numerical.js";

// Register all builtins on module load
registerMathFunctions();
registerPrngFunctions();
registerArrayFunctions();
registerArrayManipulationFunctions();
registerIntrospectionFunctions();
registerReductionFunctions();
registerStringFunctions();
registerLinearAlgebraFunctions();
registerMiscFunctions();
registerGraphicsFunctions();
registerValidatorFunctions();
registerDummyFunctions();
registerNumericalFunctions();
