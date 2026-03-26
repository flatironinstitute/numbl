export type {
  RuntimeValue,
  RuntimeNumber,
  RuntimeTensor,
  RuntimeString,
  RuntimeLogical,
  RuntimeCell,
  RuntimeStruct,
  RuntimeFunction,
  RuntimeClassInstance,
  RuntimeComplexNumber,
  RuntimeDummyHandle,
  RuntimeStructArray,
  RuntimeSparseMatrix,
} from "./types.js";

export {
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeString,
  isRuntimeChar,
  isRuntimeTensor,
  isRuntimeCell,
  isRuntimeStruct,
  isRuntimeFunction,
  isRuntimeClassInstance,
  isRuntimeComplexNumber,
  isRuntimeDummyHandle,
  isRuntimeStructArray,
  isRuntimeSparseMatrix,
  kstr,
  USE_FLOAT32,
  FloatXArray,
} from "./types.js";

export {
  RuntimeError,
  offsetToLine,
  offsetToLineFast,
  buildLineTable,
  offsetToColumn,
  extractSnippet,
} from "./error.js";
export type { CallFrame } from "./error.js";

export { runtimeError, formatError } from "./errorHelpers.js";

export {
  tensorSize2D,
  numel,
  colMajorIndex,
  ind2sub,
  sub2ind,
  shareRuntimeValue,
} from "./utils.js";

export { toNumber, toBool, toString } from "./convert.js";
export { valuesAreEqual } from "./compare.js";

export { displayValue } from "./display.js";

export { RTV } from "./constructors.js";

export {
  mAdd,
  mSub,
  mMul,
  mElemMul,
  mDiv,
  mElemDiv,
  mPow,
  mElemPow,
  mNeg,
  mTranspose,
  mConjugateTranspose,
  mEqual,
  mNotEqual,
  mLess,
  mLessEqual,
  mGreater,
  mGreaterEqual,
} from "../helpers/arithmetic.js";

export {
  COLON_INDEX,
  isColonIndex,
  indexIntoRTValue as mIndex,
  storeIntoRTValueIndex as mIndexStore,
} from "./indexing.js";
export {
  getRTValueField as mGetField,
  setRTValueField as mSetField,
} from "./struct-access.js";
export {
  makeRangeTensor as mRange,
  horzcat,
  vertcat,
} from "./tensor-construction.js";
