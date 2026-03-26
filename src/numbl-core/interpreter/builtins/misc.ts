/**
 * Misc builtins: substruct, odeset, peaks, now, datestr, lastwarn, verLessThan.
 */

import {
  FloatXArray,
  isRuntimeCell,
  isRuntimeChar,
  isRuntimeNumber,
  isRuntimeString,
} from "../../runtime/types.js";
import type { RuntimeValue, RuntimeStruct } from "../../runtime/types.js";
import { RTV, RuntimeError } from "../../runtime/index.js";
import { toNumber } from "../../runtime/convert.js";
import { registerIBuiltin } from "./types.js";

// ── substruct ────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "substruct",
  resolve: () => ({
    outputTypes: [{ kind: "unknown" }],
    apply: args => {
      if (args.length < 2 || args.length % 2 !== 0)
        throw new RuntimeError(
          "substruct requires pairs of (type, subs) arguments"
        );
      const elements: RuntimeStruct[] = [];
      for (let i = 0; i < args.length; i += 2) {
        const typeArg = args[i];
        const subsArg = args[i + 1];
        if (!isRuntimeChar(typeArg) && !isRuntimeString(typeArg))
          throw new RuntimeError("substruct: type must be a string");
        const typeStr = isRuntimeChar(typeArg)
          ? typeArg.value
          : isRuntimeString(typeArg)
            ? typeArg
            : "";
        if (typeStr !== "." && typeStr !== "()" && typeStr !== "{}")
          throw new RuntimeError(
            `substruct: type must be '.', '()', or '{}', got '${typeStr}'`
          );
        let subs: RuntimeValue;
        if (typeStr === ".") {
          subs = subsArg;
        } else {
          subs = isRuntimeCell(subsArg) ? subsArg : RTV.cell([subsArg], [1, 1]);
        }
        elements.push(RTV.struct({ type: RTV.char(typeStr), subs }));
      }
      return RTV.structArray(["type", "subs"], elements);
    },
  }),
});

// ── odeset (dummy) ───────────────────────────────────────────────────────

registerIBuiltin({
  name: "odeset",
  resolve: () => ({
    outputTypes: [{ kind: "unknown" }],
    apply: () => RTV.struct({}),
  }),
});

// ── peaks ────────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "peaks",
  resolve: () => ({
    outputTypes: [{ kind: "tensor", isComplex: false }],
    apply: args => {
      const n =
        args.length > 0
          ? isRuntimeNumber(args[0])
            ? (args[0] as number)
            : toNumber(args[0])
          : 49;
      const data = new FloatXArray(n * n);
      for (let j = 0; j < n; j++) {
        const y = -3 + (6 * j) / (n - 1);
        for (let i = 0; i < n; i++) {
          const x = -3 + (6 * i) / (n - 1);
          const x2 = x * x;
          const y2 = y * y;
          const z =
            3 * Math.pow(1 - x, 2) * Math.exp(-x2 - Math.pow(y + 1, 2)) -
            10 * (x / 5 - x * x2 - Math.pow(y, 5)) * Math.exp(-x2 - y2) -
            (1 / 3) * Math.exp(-Math.pow(x + 1, 2) - y2);
          data[j * n + i] = z;
        }
      }
      return RTV.tensor(data, [n, n]);
    },
  }),
});

// ── now ──────────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "now",
  resolve: () => ({
    outputTypes: [{ kind: "number" }],
    apply: () => {
      // MATLAB serial date number: days since Jan 0, year 0000
      // Jan 1, 1970 = MATLAB day 719529
      const matlabEpoch = 719529;
      return matlabEpoch + Date.now() / 86400000;
    },
  }),
});

// ── datestr ──────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "datestr",
  resolve: () => ({
    outputTypes: [{ kind: "char" }],
    apply: args => {
      if (args.length < 1)
        throw new RuntimeError("datestr requires at least 1 argument");
      const datenum = toNumber(args[0]);
      // Convert MATLAB serial date to JS Date
      const matlabEpoch = 719529;
      const ms = (datenum - matlabEpoch) * 86400000;
      const d = new Date(ms);
      const months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
      ];
      const day = String(d.getUTCDate()).padStart(2, "0");
      const mon = months[d.getUTCMonth()];
      const year = d.getUTCFullYear();
      const hh = String(d.getUTCHours()).padStart(2, "0");
      const mm = String(d.getUTCMinutes()).padStart(2, "0");
      const ss = String(d.getUTCSeconds()).padStart(2, "0");
      return RTV.char(`${day}-${mon}-${year} ${hh}:${mm}:${ss}`);
    },
  }),
});

// ── lastwarn ─────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "lastwarn",
  resolve: () => ({
    outputTypes: [{ kind: "char" }],
    apply: () => RTV.char(""),
  }),
});

// ── verLessThan ──────────────────────────────────────────────────────────

registerIBuiltin({
  name: "verLessThan",
  resolve: () => ({
    outputTypes: [{ kind: "boolean" }],
    apply: args => {
      if (args.length !== 2)
        throw new RuntimeError("verLessThan requires 2 arguments");
      return false; // numbl aims to be like modern MATLAB
    },
  }),
});
