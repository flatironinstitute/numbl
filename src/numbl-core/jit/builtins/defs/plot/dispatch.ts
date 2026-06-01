/**
 * Plot dispatch — every plotting builtin (`plot`, `surf`, `imagesc`,
 * `bar`, `figure`, `hold`, `xlabel`, …) shares a single lowering.
 */

import { UnsupportedConstruct } from "../../../lowering/errors.js";
import { VOID } from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import {
  emitFormatSlot,
  emitFormatSlotArray,
  validateFormatArgs,
} from "../io/_format_args.js";
import { PLOT_ALL_NAMES } from "../../../numbl/index.js";
import { mtoc2_plot_dispatch as jsPlotDispatch } from "../../runtime/snippets.gen.js";

function unwrapPlotArg(v: unknown): unknown {
  // Char wrappers flatten to plain strings for the encoder; everything
  // else passes through (tensor / number / string).
  if (typeof v === "object" && v !== null) {
    const o = v as { mtoc2Tag?: string; value?: string };
    if (o.mtoc2Tag === "char" && typeof o.value === "string") return o.value;
  }
  return v;
}

const PLOT_BUILTIN_NAMES: ReadonlyArray<string> = PLOT_ALL_NAMES;

function escapeCStringLiteral(s: string): string {
  let out = "";
  for (const ch of s) {
    if (ch === "\\" || ch === '"') {
      out += "\\" + ch;
    } else {
      out += ch;
    }
  }
  return out;
}

function makePlotBuiltin(name: string): Builtin {
  const cName = escapeCStringLiteral(name);
  return {
    name,
    transfer(argTypes, nargout) {
      if (nargout !== 1) {
        throw new UnsupportedConstruct(
          `'${name}' does not support multi-output (nargout=${nargout})`
        );
      }
      validateFormatArgs(name, argTypes, 0);
      return [VOID];
    },
    emitC({ argsC, argTypes, useRuntime }) {
      useRuntime("mtoc2_plot_dispatch");
      const slots: string[] = [];
      for (let i = 0; i < argTypes.length; i++) {
        slots.push(emitFormatSlot(name, argsC[i], argTypes[i], i));
      }
      return `mtoc2_plot_dispatch("${cName}", ${slots.length}, ${emitFormatSlotArray(slots)})`;
    },
    emitJs({ argsJs, useRuntime }) {
      useRuntime("mtoc2_plot_dispatch");
      const argList = [JSON.stringify(name), ...argsJs].join(", ");
      return `mtoc2_plot_dispatch(${argList})`;
    },
    call({ args, ctx }) {
      globalThis.$write = ctx.helpers.write;
      jsPlotDispatch(name, ...args.map(unwrapPlotArg));
      return [];
    },
  };
}

export const plotBuiltins: ReadonlyArray<Builtin> =
  PLOT_BUILTIN_NAMES.map(makePlotBuiltin);
