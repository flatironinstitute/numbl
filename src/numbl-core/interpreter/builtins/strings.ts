/**
 * Interpreter IBuiltins for string/char operations.
 *
 * Functions that involve cell arrays (strsplit, strjoin, symvar, regexp/regexpi)
 * or the parser (symvar) are left to legacy builtins since JitType has no cell kind.
 */

import type { RuntimeValue } from "../../runtime/types.js";
import {
  isRuntimeChar,
  isRuntimeCell,
  isRuntimeLogical,
  isRuntimeNumber,
  isRuntimeString,
  isRuntimeTensor,
  FloatXArray,
} from "../../runtime/types.js";
import {
  RTV,
  toNumber,
  toString,
  RuntimeError,
  displayValue,
} from "../../runtime/index.js";
import type { JitType } from "../jit/jitTypes.js";
import { registerIBuiltin } from "./types.js";
import { sprintfFormat } from "../../helpers/string.js";

// ── Type helpers ──────────────────────────────────────────────────────

function isTextType(t: JitType): boolean {
  return t.kind === "char" || t.kind === "string";
}

/** Return the same text kind as input, or null if not text. */
function preserveTextType(t: JitType): JitType | null {
  if (t.kind === "char") return { kind: "char" };
  if (t.kind === "string") return { kind: "string" };
  return null;
}

/** Resolve for a simple 1-arg text→text function that preserves char/string. */
function textPreserveResolve(fn: (s: string) => string): (
  argTypes: JitType[],
  nargout: number
) => {
  outputTypes: JitType[];
  apply: (args: RuntimeValue[]) => RuntimeValue;
} | null {
  return argTypes => {
    if (argTypes.length !== 1) return null;
    const out = preserveTextType(argTypes[0]);
    if (!out) return null;
    return {
      outputTypes: [out],
      apply: args => {
        const v = args[0];
        const s = toString(v);
        const result = fn(s);
        return isRuntimeChar(v) ? RTV.char(result) : RTV.string(result);
      },
    };
  };
}

// ── Text→text (type-preserving) ───────────────────────────────────────

registerIBuiltin({
  name: "lower",
  resolve: textPreserveResolve(s => s.toLowerCase()),
});

registerIBuiltin({
  name: "upper",
  resolve: textPreserveResolve(s => s.toUpperCase()),
});

registerIBuiltin({
  name: "strtrim",
  resolve: textPreserveResolve(s => s.trim()),
});

registerIBuiltin({
  name: "deblank",
  resolve: textPreserveResolve(s => s.replace(/\s+$/, "")),
});

registerIBuiltin({
  name: "reverse",
  resolve: textPreserveResolve(s => s.split("").reverse().join("")),
});

// ── strip (1-2 args) ─────────────────────────────────────────────────

registerIBuiltin({
  name: "strip",
  resolve: argTypes => {
    if (argTypes.length < 1 || argTypes.length > 2) return null;
    const out = preserveTextType(argTypes[0]);
    if (!out) return null;
    if (argTypes.length === 2 && !isTextType(argTypes[1])) return null;
    return {
      outputTypes: [out],
      apply: args => {
        const v = args[0];
        const s = toString(v);
        let side = "both";
        if (args.length >= 2) side = toString(args[1]).toLowerCase();
        let result: string;
        if (side === "left") result = s.replace(/^\s+/, "");
        else if (side === "right") result = s.replace(/\s+$/, "");
        else result = s.trim();
        return isRuntimeChar(v) ? RTV.char(result) : RTV.string(result);
      },
    };
  },
});

// ── pad ───────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "pad",
  resolve: argTypes => {
    if (argTypes.length < 2 || argTypes.length > 3) return null;
    const out = preserveTextType(argTypes[0]);
    if (!out) return null;
    if (argTypes[1].kind !== "number" && argTypes[1].kind !== "boolean")
      return null;
    return {
      outputTypes: [out],
      apply: args => {
        const v = args[0];
        const s = toString(v);
        const n = Math.round(toNumber(args[1]));
        let side = "right";
        if (args.length >= 3) side = toString(args[2]).toLowerCase();
        if (s.length >= n)
          return isRuntimeChar(v) ? RTV.char(s) : RTV.string(s);
        const padding = " ".repeat(n - s.length);
        const result = side === "left" ? padding + s : s + padding;
        return isRuntimeChar(v) ? RTV.char(result) : RTV.string(result);
      },
    };
  },
});

// ── erase ─────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "erase",
  resolve: argTypes => {
    if (argTypes.length !== 2) return null;
    const out = preserveTextType(argTypes[0]);
    if (!out) return null;
    if (!isTextType(argTypes[1])) return null;
    return {
      outputTypes: [out],
      apply: args => {
        const v = args[0];
        const s = toString(v);
        const pat = toString(args[1]);
        const result = s.split(pat).join("");
        return isRuntimeChar(v) ? RTV.char(result) : RTV.string(result);
      },
    };
  },
});

// ── strrep / replace ──────────────────────────────────────────────────

function strreplaceResolve(): (
  argTypes: JitType[],
  nargout: number
) => {
  outputTypes: JitType[];
  apply: (args: RuntimeValue[]) => RuntimeValue;
} | null {
  return argTypes => {
    if (argTypes.length !== 3) return null;
    const out = preserveTextType(argTypes[0]);
    if (!out) return null;
    if (!isTextType(argTypes[1]) || !isTextType(argTypes[2])) return null;
    return {
      outputTypes: [out],
      apply: args => {
        const v = args[0];
        const s = toString(v);
        const old = toString(args[1]);
        const rep = toString(args[2]);
        const result = s.split(old).join(rep);
        return isRuntimeChar(v) ? RTV.char(result) : RTV.string(result);
      },
    };
  };
}

registerIBuiltin({ name: "strrep", resolve: strreplaceResolve() });
registerIBuiltin({ name: "replace", resolve: strreplaceResolve() });

// ── regexprep ─────────────────────────────────────────────────────────

registerIBuiltin({
  name: "regexprep",
  resolve: argTypes => {
    if (argTypes.length < 3) return null;
    const out = preserveTextType(argTypes[0]);
    if (!out) return null;
    if (!isTextType(argTypes[1]) || !isTextType(argTypes[2])) return null;
    return {
      outputTypes: [out],
      apply: args => {
        const str = toString(args[0]);
        const pat = toString(args[1]);
        const rep = toString(args[2]);
        let flags = "g";
        for (let i = 3; i < args.length; i++) {
          const opt = toString(args[i]).toLowerCase();
          if (opt === "ignorecase") flags += "i";
        }
        const result = str.replace(new RegExp(pat, flags), rep);
        return isRuntimeChar(args[0]) ? RTV.char(result) : RTV.string(result);
      },
    };
  },
});

// ── extractBefore / extractAfter ──────────────────────────────────────

function extractResolve(extractFn: (s: string, arg: RuntimeValue) => string): (
  argTypes: JitType[],
  nargout: number
) => {
  outputTypes: JitType[];
  apply: (args: RuntimeValue[]) => RuntimeValue;
} | null {
  return argTypes => {
    if (argTypes.length !== 2) return null;
    const out = preserveTextType(argTypes[0]);
    if (!out) return null;
    const a1 = argTypes[1];
    if (!isTextType(a1) && a1.kind !== "number" && a1.kind !== "tensor")
      return null;
    return {
      outputTypes: [out],
      apply: args => {
        const v = args[0];
        const s = toString(v);
        const result = extractFn(s, args[1]);
        return isRuntimeChar(v) ? RTV.char(result) : RTV.string(result);
      },
    };
  };
}

registerIBuiltin({
  name: "extractBefore",
  resolve: extractResolve((s, arg) => {
    let pos: number;
    if (
      isRuntimeNumber(arg) ||
      (isRuntimeTensor(arg) && arg.data.length === 1)
    ) {
      pos = Math.round(toNumber(arg)) - 1;
    } else {
      const pat = toString(arg);
      pos = s.indexOf(pat);
      if (pos === -1)
        throw new RuntimeError("extractBefore: pattern not found");
    }
    return s.substring(0, pos);
  }),
});

registerIBuiltin({
  name: "extractAfter",
  resolve: extractResolve((s, arg) => {
    let pos: number;
    if (
      isRuntimeNumber(arg) ||
      (isRuntimeTensor(arg) && arg.data.length === 1)
    ) {
      pos = Math.round(toNumber(arg));
    } else {
      const pat = toString(arg);
      const idx = s.indexOf(pat);
      if (idx === -1) throw new RuntimeError("extractAfter: pattern not found");
      pos = idx + pat.length;
    }
    return s.substring(pos);
  }),
});

// ── extractBetween ────────────────────────────────────────────────────

registerIBuiltin({
  name: "extractBetween",
  resolve: argTypes => {
    if (argTypes.length !== 3) return null;
    const out = preserveTextType(argTypes[0]);
    if (!out) return null;
    return {
      outputTypes: [out],
      apply: args => {
        const v = args[0];
        const s = toString(v);
        const a = args[1];
        const b = args[2];
        let startPos: number;
        let endPos: number;
        if (isRuntimeNumber(a) || (isRuntimeTensor(a) && a.data.length === 1)) {
          startPos = Math.round(toNumber(a)) - 1;
        } else {
          const pat = toString(a);
          const idx = s.indexOf(pat);
          if (idx === -1)
            throw new RuntimeError("extractBetween: start pattern not found");
          startPos = idx + pat.length;
        }
        if (isRuntimeNumber(b) || (isRuntimeTensor(b) && b.data.length === 1)) {
          endPos = Math.round(toNumber(b));
        } else {
          const pat = toString(b);
          const idx = s.indexOf(pat, startPos);
          if (idx === -1)
            throw new RuntimeError("extractBetween: end pattern not found");
          endPos = idx;
        }
        const result = s.substring(startPos, endPos);
        return isRuntimeChar(v) ? RTV.char(result) : RTV.string(result);
      },
    };
  },
});

// ── insertBefore / insertAfter ────────────────────────────────────────

function insertResolve(
  insertFn: (s: string, arg: RuntimeValue, ins: string) => string
): (
  argTypes: JitType[],
  nargout: number
) => {
  outputTypes: JitType[];
  apply: (args: RuntimeValue[]) => RuntimeValue;
} | null {
  return argTypes => {
    if (argTypes.length !== 3) return null;
    const out = preserveTextType(argTypes[0]);
    if (!out) return null;
    return {
      outputTypes: [out],
      apply: args => {
        const v = args[0];
        const s = toString(v);
        const ins = toString(args[2]);
        const result = insertFn(s, args[1], ins);
        return isRuntimeChar(v) ? RTV.char(result) : RTV.string(result);
      },
    };
  };
}

registerIBuiltin({
  name: "insertBefore",
  resolve: insertResolve((s, arg, ins) => {
    let pos: number;
    if (
      isRuntimeNumber(arg) ||
      (isRuntimeTensor(arg) && arg.data.length === 1)
    ) {
      pos = Math.round(toNumber(arg)) - 1;
    } else {
      const pat = toString(arg);
      pos = s.indexOf(pat);
      if (pos === -1) throw new RuntimeError("insertBefore: pattern not found");
    }
    return s.substring(0, pos) + ins + s.substring(pos);
  }),
});

registerIBuiltin({
  name: "insertAfter",
  resolve: insertResolve((s, arg, ins) => {
    let pos: number;
    if (
      isRuntimeNumber(arg) ||
      (isRuntimeTensor(arg) && arg.data.length === 1)
    ) {
      pos = Math.round(toNumber(arg));
    } else {
      const pat = toString(arg);
      const idx = s.indexOf(pat);
      if (idx === -1) throw new RuntimeError("insertAfter: pattern not found");
      pos = idx + pat.length;
    }
    return s.substring(0, pos) + ins + s.substring(pos);
  }),
});

// ── Text→boolean functions ────────────────────────────────────────────

function isText(v: RuntimeValue): boolean {
  return isRuntimeChar(v) || isRuntimeString(v);
}

/** Element-wise strcmp helper supporting cell arrays. */
function strcmpApply(
  args: RuntimeValue[],
  cmp: (a: string, b: string) => boolean
): RuntimeValue {
  const a = args[0];
  const b = args[1];
  const aIsCell = isRuntimeCell(a);
  const bIsCell = isRuntimeCell(b);

  if (!aIsCell && !bIsCell) {
    // Scalar comparison
    if (!isText(a) || !isText(b)) return RTV.logical(false);
    return RTV.logical(cmp(toString(a), toString(b)));
  }

  // At least one cell array — element-wise comparison
  if (aIsCell && bIsCell) {
    // Both cells — must be same size
    const cellA = a as import("../../runtime/types.js").RuntimeCell;
    const cellB = b as import("../../runtime/types.js").RuntimeCell;
    const len = cellA.data.length;
    const result = new FloatXArray(len);
    for (let i = 0; i < len; i++) {
      const ai = cellA.data[i];
      const bi = cellB.data[i];
      result[i] =
        isText(ai) && isText(bi) && cmp(toString(ai), toString(bi)) ? 1 : 0;
    }
    return {
      kind: "tensor",
      data: result,
      shape: cellA.shape.slice(),
      _isLogical: true,
      _rc: 1,
    };
  }

  // One cell, one scalar
  const cell = (
    aIsCell ? a : b
  ) as import("../../runtime/types.js").RuntimeCell;
  const scalar = aIsCell ? b : a;
  const len = cell.data.length;
  const result = new FloatXArray(len);
  for (let i = 0; i < len; i++) {
    const elem = cell.data[i];
    const [s1, s2] = aIsCell ? [elem, scalar] : [scalar, elem];
    result[i] =
      isText(s1) && isText(s2) && cmp(toString(s1), toString(s2)) ? 1 : 0;
  }
  return {
    kind: "tensor",
    data: result,
    shape: cell.shape.slice(),
    _isLogical: true,
    _rc: 1,
  };
}

registerIBuiltin({
  name: "strcmp",
  resolve: argTypes => {
    if (argTypes.length !== 2) return null;
    return {
      outputTypes: [{ kind: "boolean" }],
      apply: args => strcmpApply(args, (a, b) => a === b),
    };
  },
});

registerIBuiltin({
  name: "strcmpi",
  resolve: argTypes => {
    if (argTypes.length !== 2) return null;
    return {
      outputTypes: [{ kind: "boolean" }],
      apply: args =>
        strcmpApply(args, (a, b) => a.toLowerCase() === b.toLowerCase()),
    };
  },
});

registerIBuiltin({
  name: "strncmp",
  resolve: argTypes => {
    if (argTypes.length !== 3) return null;
    return {
      outputTypes: [{ kind: "boolean" }],
      apply: args => {
        if (!isText(args[0]) || !isText(args[1])) return RTV.logical(false);
        const n = Math.round(toNumber(args[2]));
        const s1 = toString(args[0]);
        const s2 = toString(args[1]);
        return RTV.logical(s1.substring(0, n) === s2.substring(0, n));
      },
    };
  },
});

registerIBuiltin({
  name: "strncmpi",
  resolve: argTypes => {
    if (argTypes.length !== 3) return null;
    return {
      outputTypes: [{ kind: "boolean" }],
      apply: args => {
        if (!isText(args[0]) || !isText(args[1])) return RTV.logical(false);
        const n = Math.round(toNumber(args[2]));
        const s1 = toString(args[0]).toLowerCase();
        const s2 = toString(args[1]).toLowerCase();
        return RTV.logical(s1.substring(0, n) === s2.substring(0, n));
      },
    };
  },
});

registerIBuiltin({
  name: "contains",
  resolve: argTypes => {
    if (argTypes.length < 2) return null;
    if (!isTextType(argTypes[0])) return null;
    // Fall back for cell pattern arg
    if (argTypes[1].kind === "unknown") return null;
    return {
      outputTypes: [{ kind: "boolean" }],
      apply: args => {
        const s = toString(args[0]);
        const pat = args[1];
        if (isRuntimeCell(pat)) {
          for (let i = 0; i < pat.data.length; i++) {
            if (s.includes(toString(pat.data[i]))) return RTV.logical(true);
          }
          return RTV.logical(false);
        }
        return RTV.logical(s.includes(toString(pat)));
      },
    };
  },
});

registerIBuiltin({
  name: "startsWith",
  resolve: argTypes => {
    if (argTypes.length < 2) return null;
    if (!isTextType(argTypes[0]) || !isTextType(argTypes[1])) return null;
    return {
      outputTypes: [{ kind: "boolean" }],
      apply: args =>
        RTV.logical(toString(args[0]).startsWith(toString(args[1]))),
    };
  },
});

registerIBuiltin({
  name: "endsWith",
  resolve: argTypes => {
    if (argTypes.length < 2) return null;
    if (!isTextType(argTypes[0]) || !isTextType(argTypes[1])) return null;
    return {
      outputTypes: [{ kind: "boolean" }],
      apply: args => RTV.logical(toString(args[0]).endsWith(toString(args[1]))),
    };
  },
});

// ── Text→number functions ─────────────────────────────────────────────

registerIBuiltin({
  name: "strlength",
  resolve: argTypes => {
    if (argTypes.length !== 1) return null;
    if (!isTextType(argTypes[0])) return null;
    return {
      outputTypes: [{ kind: "number", sign: "nonneg" }],
      apply: args => {
        const v = args[0];
        if (isRuntimeString(v)) return RTV.num(v.length);
        if (isRuntimeChar(v)) return RTV.num(v.value.length);
        throw new RuntimeError("strlength: argument must be a string or char");
      },
    };
  },
});

registerIBuiltin({
  name: "str2double",
  resolve: argTypes => {
    if (argTypes.length !== 1) return null;
    if (!isTextType(argTypes[0])) return null;
    return {
      outputTypes: [{ kind: "number" }],
      apply: args => {
        const s = toString(args[0]).trim();
        return RTV.num(s === "" ? NaN : Number(s));
      },
    };
  },
});

registerIBuiltin({
  name: "str2num",
  resolve: argTypes => {
    if (argTypes.length !== 1) return null;
    if (!isTextType(argTypes[0])) return null;
    // Output can be number or empty tensor — conservatively say number
    return {
      outputTypes: [{ kind: "number" }],
      apply: args => {
        const s = toString(args[0]);
        const n = Number(s);
        return Number.isNaN(n)
          ? RTV.tensor(new FloatXArray(0), [0, 0])
          : RTV.num(n);
      },
    };
  },
});

registerIBuiltin({
  name: "count",
  resolve: argTypes => {
    if (argTypes.length < 2) return null;
    if (!isTextType(argTypes[0]) || !isTextType(argTypes[1])) return null;
    return {
      outputTypes: [{ kind: "number", sign: "nonneg" }],
      apply: args => {
        const s = toString(args[0]);
        const pat = toString(args[1]);
        if (pat.length === 0) return RTV.num(s.length + 1);
        let n = 0;
        let idx = 0;
        while ((idx = s.indexOf(pat, idx)) !== -1) {
          n++;
          idx += pat.length;
        }
        return RTV.num(n);
      },
    };
  },
});

registerIBuiltin({
  name: "hex2dec",
  resolve: argTypes => {
    if (argTypes.length !== 1) return null;
    if (!isTextType(argTypes[0])) return null;
    return {
      outputTypes: [{ kind: "number" }],
      apply: args => RTV.num(parseInt(toString(args[0]), 16)),
    };
  },
});

registerIBuiltin({
  name: "bin2dec",
  resolve: argTypes => {
    if (argTypes.length !== 1) return null;
    if (!isTextType(argTypes[0])) return null;
    return {
      outputTypes: [{ kind: "number" }],
      apply: args => RTV.num(parseInt(toString(args[0]), 2)),
    };
  },
});

registerIBuiltin({
  name: "strfind",
  resolve: argTypes => {
    if (argTypes.length !== 2) return null;
    if (!isTextType(argTypes[0]) || !isTextType(argTypes[1])) return null;
    return {
      outputTypes: [{ kind: "tensor", isComplex: false, ndim: 2 }],
      apply: args => {
        const s = toString(args[0]);
        const pattern = toString(args[1]);
        const indices: number[] = [];
        let pos = 0;
        while (pos <= s.length - pattern.length) {
          const idx = s.indexOf(pattern, pos);
          if (idx === -1) break;
          indices.push(idx + 1);
          pos = idx + 1;
        }
        if (indices.length === 0) {
          return RTV.tensor(new FloatXArray(0), [1, 0]);
        }
        const data = new FloatXArray(indices.length);
        indices.forEach((v, i) => (data[i] = v));
        return RTV.tensor(data, [1, indices.length]);
      },
    };
  },
});

// ── Number/tensor→char functions ──────────────────────────────────────

registerIBuiltin({
  name: "blanks",
  resolve: argTypes => {
    if (argTypes.length !== 1) return null;
    if (argTypes[0].kind !== "number" && argTypes[0].kind !== "boolean")
      return null;
    return {
      outputTypes: [{ kind: "char" }],
      apply: args => {
        const n = Math.round(toNumber(args[0]));
        return RTV.char(" ".repeat(Math.max(0, n)));
      },
    };
  },
});

registerIBuiltin({
  name: "dec2hex",
  resolve: argTypes => {
    if (argTypes.length < 1 || argTypes.length > 2) return null;
    if (argTypes[0].kind !== "number" && argTypes[0].kind !== "boolean")
      return null;
    return {
      outputTypes: [{ kind: "char" }],
      apply: args => {
        let n = Math.round(toNumber(args[0]));
        if (n < 0) {
          // Two's complement: find smallest byte-aligned width
          let bits = 8;
          while (bits <= 52 && -(1 << (bits - 1)) > n) bits += 8;
          n = (1 << bits) + n;
        }
        let hex = n.toString(16).toUpperCase();
        if (args.length === 2) {
          const minDigits = Math.round(toNumber(args[1]));
          while (hex.length < minDigits) hex = "0" + hex;
        }
        return RTV.char(hex);
      },
    };
  },
});

registerIBuiltin({
  name: "dec2bin",
  resolve: argTypes => {
    if (argTypes.length < 1 || argTypes.length > 2) return null;
    if (argTypes[0].kind !== "number" && argTypes[0].kind !== "boolean")
      return null;
    return {
      outputTypes: [{ kind: "char" }],
      apply: args => {
        let n = Math.round(toNumber(args[0]));
        if (n < 0) {
          let bits = 8;
          while (bits <= 52 && -(1 << (bits - 1)) > n) bits += 8;
          n = (1 << bits) + n;
        }
        let bin = n.toString(2);
        if (args.length === 2) {
          const minDigits = Math.round(toNumber(args[1]));
          while (bin.length < minDigits) bin = "0" + bin;
        }
        return RTV.char(bin);
      },
    };
  },
});

registerIBuiltin({
  name: "int2str",
  resolve: argTypes => {
    if (argTypes.length !== 1) return null;
    const a = argTypes[0];
    if (a.kind !== "number" && a.kind !== "boolean" && a.kind !== "tensor")
      return null;
    return {
      outputTypes: [{ kind: "char" }],
      apply: args => {
        const v = args[0];
        const roundInt = (x: number) => Math.sign(x) * Math.round(Math.abs(x));
        if (isRuntimeTensor(v)) {
          const rows = v.shape[0] || 1;
          const cols = v.shape.length >= 2 ? v.shape[1] : v.data.length;
          const rowStrs: string[] = [];
          for (let r = 0; r < rows; r++) {
            const vals: string[] = [];
            for (let c = 0; c < cols; c++) {
              vals.push(String(roundInt(v.data[r + c * rows])));
            }
            rowStrs.push(vals.join("  "));
          }
          return RTV.char(rowStrs.join("\n"));
        }
        return RTV.char(String(roundInt(toNumber(v))));
      },
    };
  },
});

// ── num2str ───────────────────────────────────────────────────────────

function numStr(n: number): string {
  if (n === Infinity) return "Inf";
  if (n === -Infinity) return "-Inf";
  if (isNaN(n)) return "NaN";
  if (n === 0) return "0";
  const prec = 5;
  const exp = Math.floor(Math.log10(Math.abs(n)));
  let s: string;
  if (exp < -4 || exp >= prec) {
    s = n.toExponential(prec - 1);
    const ePos = s.indexOf("e");
    let mantissa = s.slice(0, ePos);
    const expPart0 = s.slice(ePos);
    if (mantissa.includes(".")) mantissa = mantissa.replace(/\.?0+$/, "");
    const expPart = expPart0.replace(/([eE][+-])(\d)$/, "$1" + "0$2");
    s = mantissa + expPart;
  } else {
    if (Number.isInteger(n)) return String(n);
    s = n.toPrecision(prec);
    if (s.includes(".")) s = s.replace(/\.?0+$/, "");
  }
  return s;
}

registerIBuiltin({
  name: "num2str",
  resolve: argTypes => {
    if (argTypes.length < 1 || argTypes.length > 2) return null;
    const a = argTypes[0];
    if (a.kind !== "number" && a.kind !== "boolean" && a.kind !== "tensor")
      return null;
    // Second arg can be format string or precision number
    if (
      argTypes.length === 2 &&
      !isTextType(argTypes[1]) &&
      argTypes[1].kind !== "number"
    )
      return null;
    return {
      outputTypes: [{ kind: "char" }],
      apply: args => {
        const v = args[0];
        if (
          args.length >= 2 &&
          (isRuntimeString(args[1]) || isRuntimeChar(args[1]))
        ) {
          return RTV.char(
            sprintfFormat(isRuntimeString(args[1]) ? args[1] : args[1].value, [
              v,
            ])
          );
        }
        const precision =
          args.length >= 2 && isRuntimeNumber(args[1])
            ? (args[1] as number)
            : undefined;
        const fmtNum = (x: number): string => {
          if (precision !== undefined) {
            let s = x.toPrecision(precision);
            if (s.includes(".")) {
              const eIdx = s.search(/[eE]/);
              if (eIdx === -1) {
                s = s.replace(/\.?0+$/, "");
              } else {
                const mantissa = s.slice(0, eIdx).replace(/\.?0+$/, "");
                let expPart = s.slice(eIdx);
                expPart = expPart.replace(/([eE][+-])(\d)$/, "$1" + "0$2");
                s = mantissa + expPart;
              }
            }
            return s;
          }
          return numStr(x);
        };
        if (isRuntimeNumber(v) || isRuntimeLogical(v)) {
          return RTV.char(fmtNum(toNumber(v)));
        }
        if (isRuntimeTensor(v)) {
          const nRows = v.shape[0] || 1;
          const nCols = v.data.length / nRows;
          const strs: string[][] = [];
          for (let r = 0; r < nRows; r++) {
            const row: string[] = [];
            for (let c = 0; c < nCols; c++) {
              row.push(fmtNum(v.data[c * nRows + r]));
            }
            strs.push(row);
          }
          const colWidths: number[] = [];
          for (let c = 0; c < nCols; c++) {
            let maxW = 0;
            for (let r = 0; r < nRows; r++) {
              if (strs[r][c].length > maxW) maxW = strs[r][c].length;
            }
            colWidths.push(maxW);
          }
          const rowStrs = strs.map(row =>
            row.map((s, c) => s.padStart(colWidths[c])).join("  ")
          );
          if (nRows === 1) return RTV.char(rowStrs[0]);
          const rowWidth = rowStrs[0].length;
          const paddedRows = rowStrs.map(r => r.padEnd(rowWidth));
          return {
            kind: "char" as const,
            value: paddedRows.join(""),
            shape: [nRows, rowWidth],
          };
        }
        return RTV.char(String(toNumber(v)));
      },
    };
  },
});

// ── mat2str ───────────────────────────────────────────────────────────

registerIBuiltin({
  name: "mat2str",
  resolve: argTypes => {
    if (argTypes.length < 1 || argTypes.length > 2) return null;
    const a = argTypes[0];
    if (a.kind !== "number" && a.kind !== "boolean" && a.kind !== "tensor")
      return null;
    return {
      outputTypes: [{ kind: "char" }],
      apply: args => {
        const v = args[0];
        const prec = args.length >= 2 ? Math.round(toNumber(args[1])) : -1;
        const fmt = (n: number) =>
          prec >= 0 ? Number(n.toPrecision(prec)).toString() : String(n);
        if (isRuntimeNumber(v)) return RTV.char(fmt(v));
        if (isRuntimeTensor(v)) {
          const nRows = v.shape[0] || 1;
          const nCols = v.data.length / nRows;
          if (nRows === 1 && nCols === 1) return RTV.char(fmt(v.data[0]));
          const rows: string[] = [];
          for (let r = 0; r < nRows; r++) {
            const elems: string[] = [];
            for (let c = 0; c < nCols; c++) {
              elems.push(fmt(v.data[c * nRows + r]));
            }
            rows.push(elems.join(" "));
          }
          return RTV.char("[" + rows.join(";") + "]");
        }
        return RTV.char(String(toNumber(v)));
      },
    };
  },
});

// ── Type conversion: char, string ─────────────────────────────────────

registerIBuiltin({
  name: "char",
  resolve: argTypes => {
    if (argTypes.length !== 1) return null;
    const a = argTypes[0];
    if (
      a.kind !== "char" &&
      a.kind !== "string" &&
      a.kind !== "number" &&
      a.kind !== "tensor"
    )
      return null;
    return {
      outputTypes: [{ kind: "char" }],
      apply: args => {
        const v = args[0];
        if (isRuntimeChar(v)) return v;
        if (isRuntimeString(v)) return RTV.char(v);
        if (isRuntimeNumber(v))
          return RTV.char(String.fromCharCode(Math.round(v)));
        if (isRuntimeTensor(v)) {
          const chars: string[] = [];
          for (let i = 0; i < v.data.length; i++) {
            chars.push(String.fromCharCode(Math.round(v.data[i])));
          }
          return RTV.char(chars.join(""));
        }
        throw new RuntimeError("char: unsupported arguments");
      },
    };
  },
});

registerIBuiltin({
  name: "string",
  resolve: argTypes => {
    if (argTypes.length !== 1) return null;
    const a = argTypes[0];
    if (a.kind === "unknown") return null;
    return {
      outputTypes: [{ kind: "string" }],
      apply: args => {
        const v = args[0];
        if (isRuntimeString(v)) return v;
        if (isRuntimeChar(v)) return RTV.string(v.value);
        return RTV.string(displayValue(v));
      },
    };
  },
});

// ── strcat ────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "strcat",
  resolve: argTypes => {
    if (argTypes.length < 1) return null;
    // Fall back for any unknown/cell args
    if (argTypes.some(t => t.kind === "unknown")) return null;
    // Must have at least one text arg
    if (!argTypes.some(t => isTextType(t))) return null;
    // If any string arg, output is string; otherwise char
    const hasString = argTypes.some(t => t.kind === "string");
    const outType: JitType = hasString ? { kind: "string" } : { kind: "char" };
    return {
      outputTypes: [outType],
      apply: args => {
        const anyString = args.some(a => isRuntimeString(a));
        if (anyString) {
          return RTV.string(args.map(a => toString(a)).join(""));
        }
        const parts = args.map(a => toString(a).replace(/[ \t\v\n\r\f]+$/, ""));
        return RTV.char(parts.join(""));
      },
    };
  },
});

// ── sprintf ───────────────────────────────────────────────────────────

registerIBuiltin({
  name: "sprintf",
  resolve: argTypes => {
    if (argTypes.length < 1) return null;
    if (!isTextType(argTypes[0])) return null;
    // Output type matches format arg type
    const outType: JitType =
      argTypes[0].kind === "char" ? { kind: "char" } : { kind: "string" };
    return {
      outputTypes: [outType],
      apply: args => {
        const fmtArg = args[0];
        const fmt = toString(fmtArg);
        const result = sprintfFormat(fmt, args.slice(1));
        return isRuntimeChar(fmtArg) ? RTV.char(result) : RTV.string(result);
      },
    };
  },
});

// ── sscanf ────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "sscanf",
  resolve: (argTypes, nargout) => {
    if (argTypes.length < 2) return null;
    if (!isTextType(argTypes[0]) || !isTextType(argTypes[1])) return null;
    const outputTypes: JitType[] = [{ kind: "number" }];
    if (nargout >= 2) outputTypes.push({ kind: "number" });
    return {
      outputTypes,
      apply: (args, nout) => {
        const str = toString(args[0]);
        const fmt = toString(args[1]);
        const maxCount = args.length >= 3 ? toNumber(args[2]) : Infinity;
        const results: number[] = [];
        let strPos = 0;
        let fmtPos = 0;

        while (
          fmtPos < fmt.length &&
          strPos < str.length &&
          results.length < maxCount
        ) {
          if (fmt[fmtPos] === "%") {
            fmtPos++;
            if (fmtPos >= fmt.length) break;
            const spec = fmt[fmtPos];
            fmtPos++;
            if (spec !== "c" && spec !== "s") {
              while (strPos < str.length && /\s/.test(str[strPos])) strPos++;
            }
            if (spec === "d" || spec === "i") {
              const m = str.slice(strPos).match(/^[+-]?\d+/);
              if (!m) break;
              results.push(parseInt(m[0], 10));
              strPos += m[0].length;
            } else if (spec === "f" || spec === "e" || spec === "g") {
              const m = str
                .slice(strPos)
                .match(/^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?/);
              if (!m) break;
              results.push(parseFloat(m[0]));
              strPos += m[0].length;
            } else if (spec === "x") {
              const m = str.slice(strPos).match(/^[+-]?[0-9a-fA-F]+/);
              if (!m) break;
              results.push(parseInt(m[0], 16));
              strPos += m[0].length;
            } else if (spec === "o") {
              const m = str.slice(strPos).match(/^[+-]?[0-7]+/);
              if (!m) break;
              results.push(parseInt(m[0], 8));
              strPos += m[0].length;
            } else if (spec === "c") {
              results.push(str.charCodeAt(strPos));
              strPos++;
            } else if (spec === "s") {
              const m = str.slice(strPos).match(/^\S+/);
              if (!m) break;
              for (
                let ci = 0;
                ci < m[0].length && results.length < maxCount;
                ci++
              ) {
                results.push(m[0].charCodeAt(ci));
              }
              strPos += m[0].length;
            }
          } else if (/\s/.test(fmt[fmtPos])) {
            fmtPos++;
            while (strPos < str.length && /\s/.test(str[strPos])) strPos++;
          } else {
            if (str[strPos] !== fmt[fmtPos]) break;
            strPos++;
            fmtPos++;
          }
          if (
            fmtPos >= fmt.length &&
            results.length < maxCount &&
            strPos < str.length
          ) {
            fmtPos = 0;
          }
        }

        const vals =
          results.length === 1
            ? RTV.num(results[0])
            : RTV.tensor(new FloatXArray(results), [results.length, 1]);
        if (nout >= 2) {
          return [vals, RTV.num(results.length)];
        }
        return vals;
      },
    };
  },
});

// ── strtok ────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "strtok",
  resolve: (argTypes, nargout) => {
    if (argTypes.length < 1 || argTypes.length > 2) return null;
    if (!isTextType(argTypes[0])) return null;
    if (argTypes.length === 2 && !isTextType(argTypes[1])) return null;
    const outputTypes: JitType[] = [{ kind: "char" }];
    if (nargout >= 2) outputTypes.push({ kind: "char" });
    return {
      outputTypes,
      apply: (args, nout) => {
        const s = toString(args[0]);
        const delims = args.length >= 2 ? toString(args[1]) : " \t\n\r\f\v";
        let start = 0;
        while (start < s.length && delims.includes(s[start])) start++;
        if (start >= s.length) {
          if (nout <= 1) return RTV.char("");
          return [RTV.char(""), RTV.char("")];
        }
        let end = start;
        while (end < s.length && !delims.includes(s[end])) end++;
        const token = s.substring(start, end);
        if (nout <= 1) return RTV.char(token);
        return [RTV.char(token), RTV.char(s.substring(end))];
      },
    };
  },
});
