export type {
  RuntimeValue,
  RuntimeTensor,
  RuntimeString,
  RuntimeLogical,
  RuntimeCell,
  RuntimeStruct,
  RuntimeFunction,
  RuntimeDictionary,
} from "./types.js";

export { RuntimeError, offsetToLine, offsetToColumn } from "./error.js";
export type { CallFrame } from "./error.js";

export {
  tensorSize2D,
  numel,
  colMajorIndex,
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
