import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import { isText, typeToString } from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import {
  emitFormatSlot,
  emitFormatSlotArray,
  emitTextView,
  validateFormatArgs,
} from "./_format_args.js";
import {
  mtoc2_sprintf_str as jsSprintfStr,
  mtoc2_sprintf_char as jsSprintfChar,
} from "../../runtime/snippets.gen.js";

function unwrapFmtArg(v: unknown): unknown {
  if (typeof v === "object" && v !== null) {
    const o = v as { mtoc2Tag?: string; value?: string };
    if (o.mtoc2Tag === "char" && typeof o.value === "string") return o.value;
  }
  return v;
}

export const sprintfBuiltin: Builtin = {
  name: "sprintf",
  transfer(argTypes, nargout) {
    if (argTypes.length === 0) {
      throw new TypeError(`'sprintf' requires at least a format arg`);
    }
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'sprintf' does not support multi-output (nargout=${nargout})`
      );
    }
    const fmt = argTypes[0];
    if (!isText(fmt)) {
      throw new TypeError(
        `'sprintf' first arg must be char or string (got ${typeToString(fmt)})`
      );
    }
    validateFormatArgs("sprintf", argTypes, 1);
    return [fmt.kind === "String" ? { kind: "String" } : { kind: "Char" }];
  },
  emitC({ argsC, argTypes, useRuntime }) {
    useRuntime("mtoc2_sprintf_str");
    useRuntime("mtoc2_sprintf_char");
    const fmt = argTypes[0];
    const fmtView = emitTextView(argsC[0], fmt);
    const slots: string[] = [];
    for (let i = 1; i < argTypes.length; i++) {
      slots.push(emitFormatSlot("sprintf", argsC[i], argTypes[i], i));
    }
    const fn =
      fmt.kind === "String" ? "mtoc2_sprintf_str" : "mtoc2_sprintf_char";
    return `${fn}(${fmtView}, ${slots.length}, ${emitFormatSlotArray(slots)})`;
  },
  emitJs({ argsJs, argTypes, useRuntime }) {
    const fmt = argTypes[0];
    useRuntime(
      fmt.kind === "String" ? "mtoc2_sprintf_str" : "mtoc2_sprintf_char"
    );
    const fn =
      fmt.kind === "String" ? "mtoc2_sprintf_str" : "mtoc2_sprintf_char";
    return `${fn}(${argsJs.join(", ")})`;
  },
  call({ args, argTypes }) {
    const unwrapped = args.map(unwrapFmtArg);
    const fmt = unwrapped[0] as string;
    const rest = unwrapped.slice(1);
    if (argTypes[0].kind === "String") {
      return [jsSprintfStr(fmt, ...rest)];
    }
    return [
      jsSprintfChar(
        fmt,
        ...rest
      ) as unknown as import("../../../runtime/value.js").RuntimeChar,
    ];
  },
};
