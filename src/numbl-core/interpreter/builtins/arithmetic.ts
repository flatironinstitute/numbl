/**
 * Binary scalar builtins: atan2, min, max, mod, rem, power.
 */

import {
  registerIBuiltin,
  binaryNumberOnly,
  applyBinaryScalar,
} from "./types.js";

registerIBuiltin({
  name: "atan2",
  typeRule: argTypes => binaryNumberOnly(argTypes),
  apply: args => applyBinaryScalar(args, Math.atan2, "atan2"),
});

registerIBuiltin({
  name: "min",
  typeRule: argTypes => {
    if (argTypes.length !== 2) return null;
    if (argTypes[0].kind !== "number" || argTypes[1].kind !== "number")
      return null;
    return [
      {
        kind: "number",
        nonneg: !!argTypes[0].nonneg && !!argTypes[1].nonneg,
      },
    ];
  },
  apply: args => applyBinaryScalar(args, Math.min, "min"),
});

registerIBuiltin({
  name: "max",
  typeRule: argTypes => {
    if (argTypes.length !== 2) return null;
    if (argTypes[0].kind !== "number" || argTypes[1].kind !== "number")
      return null;
    return [
      {
        kind: "number",
        nonneg: !!argTypes[0].nonneg && !!argTypes[1].nonneg,
      },
    ];
  },
  apply: args => applyBinaryScalar(args, Math.max, "max"),
});

registerIBuiltin({
  name: "mod",
  typeRule: argTypes => {
    if (argTypes.length !== 2) return null;
    if (argTypes[0].kind !== "number" || argTypes[1].kind !== "number")
      return null;
    return [{ kind: "number", nonneg: !!argTypes[1].nonneg }];
  },
  apply: args => applyBinaryScalar(args, (a, b) => ((a % b) + b) % b, "mod"),
});

registerIBuiltin({
  name: "rem",
  typeRule: argTypes => binaryNumberOnly(argTypes),
  apply: args => applyBinaryScalar(args, (a, b) => a % b, "rem"),
});

registerIBuiltin({
  name: "power",
  typeRule: argTypes => binaryNumberOnly(argTypes),
  apply: args => applyBinaryScalar(args, Math.pow, "power"),
});
