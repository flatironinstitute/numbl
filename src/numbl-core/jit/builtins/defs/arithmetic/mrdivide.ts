import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import { isMultiElement } from "../../../lowering/types.js";
import {
  type Builtin,
  requireEmitC,
  requireEmitJs,
  requireCall,
} from "../../registry.js";
import { requireRealOrComplex } from "../_shared.js";
import { rdivide } from "./rdivide.js";

/** `mrdivide` (matrix /): mirrors `rdivide` when at least one arg is
 *  scalar; rejects the both-tensor case until matrix right-division
 *  is implemented. */
export const mrdivide: Builtin = {
  name: "mrdivide",
  transfer(argTypes, nargout) {
    if (argTypes.length !== 2) {
      throw new TypeError(
        `'mrdivide' expects 2 arg(s), got ${argTypes.length}`
      );
    }
    const a = argTypes[0];
    const b = argTypes[1];
    requireRealOrComplex(a, `'mrdivide' arg 1`);
    requireRealOrComplex(b, `'mrdivide' arg 2`);
    if (isMultiElement(a) && isMultiElement(b)) {
      throw new UnsupportedConstruct(
        `matrix right-division (a/b on two tensors) is not yet supported; use './' for elementwise`
      );
    }
    return rdivide.transfer(argTypes, nargout);
  },
  emitC(args) {
    return requireEmitC(rdivide)(args);
  },
  emitJs(args) {
    return requireEmitJs(rdivide)(args);
  },
  call(args) {
    return requireCall(rdivide)(args);
  },
  // `mrdivide` between two tensors is rejected in `transfer`; the only
  // case reaching emit is at-least-one-scalar — identical to elementwise
  // `rdivide`. Mark elementwise so the fused emitter can render
  // `t / scalar` as one inline loop.
  elementwise: true,
};
