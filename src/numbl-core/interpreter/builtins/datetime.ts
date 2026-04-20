/**
 * datetime / duration builtins — scalar-only initial implementation.
 *
 * datetime:
 *   datetime()                         current time
 *   datetime('now' | 'today' | 'yesterday' | 'tomorrow')
 *   datetime(Y, M, D)
 *   datetime(Y, M, D, H, MI, S[, MS])
 *   datetime(X, 'ConvertFrom', 'datenum' | 'posixtime' | 'excel' | 'excel1904')
 *
 * datetime values are class_instance with className="datetime" and fields
 * Year/Month/Day/Hour/Minute/Second. They are display-formatted by
 * display.ts as "dd-MMM-yyyy [HH:mm:ss]".
 *
 * duration:
 *   Produced by datetime - datetime, or by seconds(N) / minutes(N) / ...
 *   Represented as class_instance with className="duration" and a single
 *   Seconds field. Display format "hh:mm:ss".
 *
 * Arithmetic:
 *   datetime - datetime -> duration
 *   datetime + duration -> datetime
 *   datetime - duration -> datetime
 *   duration + duration -> duration
 *   duration - duration -> duration
 *
 * `seconds(d)` returns the numeric seconds of a duration, or wraps a
 * number as a duration.
 */

import type {
  RuntimeValue,
  RuntimeClassInstance,
} from "../../runtime/types.js";
import {
  isRuntimeChar,
  isRuntimeClassInstance,
  isRuntimeString,
} from "../../runtime/types.js";
import { RTV, RuntimeError } from "../../runtime/index.js";
import { toNumber, toString } from "../../runtime/convert.js";
import { registerIBuiltin } from "./types.js";

// MATLAB serial date number for Jan 1, 1970 00:00:00 UTC.
const MATLAB_EPOCH_DAYS = 719529;

// ── datetime construction ──────────────────────────────────────────────────

export function makeDatetime(
  year: number,
  month: number,
  day: number,
  hour: number,
  minute: number,
  second: number
): RuntimeClassInstance {
  const fields = new Map<string, RuntimeValue>([
    ["Year", RTV.num(year)],
    ["Month", RTV.num(month)],
    ["Day", RTV.num(day)],
    ["Hour", RTV.num(hour)],
    ["Minute", RTV.num(minute)],
    ["Second", RTV.num(second)],
  ]);
  return {
    kind: "class_instance",
    className: "datetime",
    fields,
    isHandleClass: false,
  };
}

function datetimeFromDate(d: Date): RuntimeClassInstance {
  return makeDatetime(
    d.getFullYear(),
    d.getMonth() + 1,
    d.getDate(),
    d.getHours(),
    d.getMinutes(),
    d.getSeconds() + d.getMilliseconds() / 1000
  );
}

// For calendar-based conversions (datenum, excel, ...), compute the
// Y/M/D/H/MI/S directly in UTC so the result is a pure calendar value
// that doesn't shift with the host system's local time zone. This
// matches MATLAB: `datetime(739252, 'ConvertFrom', 'datenum')` is
// 01-Jan-2024 regardless of where the machine runs.
function datetimeFromUtcMs(ms: number): RuntimeClassInstance {
  const d = new Date(ms);
  return makeDatetime(
    d.getUTCFullYear(),
    d.getUTCMonth() + 1,
    d.getUTCDate(),
    d.getUTCHours(),
    d.getUTCMinutes(),
    d.getUTCSeconds() + d.getUTCMilliseconds() / 1000
  );
}

function datetimeFromDatenum(dn: number): RuntimeClassInstance {
  return datetimeFromUtcMs((dn - MATLAB_EPOCH_DAYS) * 86400000);
}

// ── duration construction / introspection ─────────────────────────────────

export function makeDuration(totalSeconds: number): RuntimeClassInstance {
  const fields = new Map<string, RuntimeValue>([
    ["Seconds", RTV.num(totalSeconds)],
  ]);
  return {
    kind: "class_instance",
    className: "duration",
    fields,
    isHandleClass: false,
  };
}

function durationSeconds(v: RuntimeClassInstance): number {
  const s = v.fields.get("Seconds");
  if (typeof s === "number") return s;
  if (typeof s === "boolean") return s ? 1 : 0;
  throw new RuntimeError("duration: invalid Seconds field");
}

// ── datetime ⇄ numeric conversion ──────────────────────────────────────────

function fieldNum(v: RuntimeClassInstance, name: string, dflt: number): number {
  const fv = v.fields.get(name);
  if (fv === undefined) return dflt;
  if (typeof fv === "number") return fv;
  if (typeof fv === "boolean") return fv ? 1 : 0;
  return dflt;
}

/**
 * Treat a datetime as a pure calendar value (Y/M/D/H/MI/S) and produce
 * "UTC ms" for it — the same treatment used on the way in by
 * `datetimeFromUtcMs`. This keeps `datetime - datetime` symmetric with
 * the constructors: calendar-in, seconds-out, no local-TZ drift.
 */
function datetimeToUtcMs(v: RuntimeClassInstance): number {
  const year = fieldNum(v, "Year", 0);
  const month = fieldNum(v, "Month", 1);
  const day = fieldNum(v, "Day", 1);
  const hour = fieldNum(v, "Hour", 0);
  const minute = fieldNum(v, "Minute", 0);
  const second = fieldNum(v, "Second", 0);
  const wholeSec = Math.floor(second);
  const ms = Math.round((second - wholeSec) * 1000);
  return Date.UTC(year, month - 1, day, hour, minute, wholeSec, ms);
}

// ── argument helpers ───────────────────────────────────────────────────────

function argToString(v: RuntimeValue): string | null {
  if (isRuntimeChar(v)) return toString(v);
  if (isRuntimeString(v)) return v as string;
  return null;
}

function isDatetime(v: RuntimeValue): v is RuntimeClassInstance {
  return isRuntimeClassInstance(v) && v.className === "datetime";
}

function isDuration(v: RuntimeValue): v is RuntimeClassInstance {
  return isRuntimeClassInstance(v) && v.className === "duration";
}

// ── datetime constructor ───────────────────────────────────────────────────

registerIBuiltin({
  name: "datetime",
  help: {
    signatures: [
      "t = datetime()",
      "t = datetime('now' | 'today' | 'yesterday' | 'tomorrow')",
      "t = datetime(Y, M, D)",
      "t = datetime(Y, M, D, H, MI, S)",
      "t = datetime(X, 'ConvertFrom', dateType)",
    ],
    description:
      "Create a datetime value. Supported dateType values for ConvertFrom: " +
      "'datenum', 'posixtime', 'excel', 'excel1904'.",
  },
  resolve: () => ({
    outputTypes: [
      {
        kind: "class_instance",
        className: "datetime",
        isHandleClass: false,
        fields: {},
      },
    ],
    apply: args => {
      if (args.length === 0) {
        return datetimeFromDate(new Date());
      }

      // datetime('now' | 'today' | 'yesterday' | 'tomorrow')
      if (args.length === 1) {
        const s = argToString(args[0]);
        if (s !== null) {
          const now = new Date();
          const lc = s.toLowerCase();
          if (lc === "now") return datetimeFromDate(now);
          if (lc === "today") {
            return makeDatetime(
              now.getFullYear(),
              now.getMonth() + 1,
              now.getDate(),
              0,
              0,
              0
            );
          }
          if (lc === "yesterday" || lc === "tomorrow") {
            const d = new Date(
              now.getFullYear(),
              now.getMonth(),
              now.getDate() + (lc === "tomorrow" ? 1 : -1)
            );
            return makeDatetime(
              d.getFullYear(),
              d.getMonth() + 1,
              d.getDate(),
              0,
              0,
              0
            );
          }
          throw new RuntimeError(
            `datetime: unrecognized relative-day string '${s}'`
          );
        }
      }

      // datetime(X, 'ConvertFrom', dateType)
      if (args.length === 3) {
        const flag = argToString(args[1]);
        const kind = argToString(args[2]);
        if (flag !== null && flag.toLowerCase() === "convertfrom") {
          if (kind === null)
            throw new RuntimeError("datetime: dateType must be text");
          const x = toNumber(args[0]);
          const k = kind.toLowerCase();
          if (k === "datenum") return datetimeFromDatenum(x);
          if (k === "posixtime") return datetimeFromUtcMs(x * 1000);
          if (k === "excel") {
            return datetimeFromUtcMs((x - 25569) * 86400000);
          }
          if (k === "excel1904") {
            return datetimeFromUtcMs((x - 24107) * 86400000);
          }
          throw new RuntimeError(
            `datetime: unsupported ConvertFrom type '${kind}'`
          );
        }
      }

      // datetime(Y, M, D [, H, MI, S [, MS]])
      if (args.length === 3 || args.length === 6 || args.length === 7) {
        const Y = toNumber(args[0]);
        const M = toNumber(args[1]);
        const D = toNumber(args[2]);
        const H = args.length >= 6 ? toNumber(args[3]) : 0;
        const MI = args.length >= 6 ? toNumber(args[4]) : 0;
        const S = args.length >= 6 ? toNumber(args[5]) : 0;
        const MS = args.length >= 7 ? toNumber(args[6]) : 0;
        return makeDatetime(Y, M, D, H, MI, S + MS / 1000);
      }

      throw new RuntimeError(
        `datetime: unsupported argument pattern (nargin=${args.length})`
      );
    },
  }),
});

// ── duration constructor-like builtins: seconds / minutes / hours / days ───
//
// Dual behavior matching MATLAB:
//   seconds(N) where N is numeric  -> a duration of N seconds
//   seconds(D) where D is duration -> numeric total seconds

type DurationUnit = {
  name: string;
  factorToSeconds: number;
};

const DURATION_UNITS: DurationUnit[] = [
  { name: "seconds", factorToSeconds: 1 },
  { name: "minutes", factorToSeconds: 60 },
  { name: "hours", factorToSeconds: 3600 },
  { name: "days", factorToSeconds: 86400 },
];

for (const unit of DURATION_UNITS) {
  registerIBuiltin({
    name: unit.name,
    help: {
      signatures: [`${unit.name}(N)`, `${unit.name}(D)`],
      description:
        `If N is numeric, create a duration of N ${unit.name}. ` +
        `If D is a duration, return its total number of ${unit.name} as a double.`,
    },
    resolve: () => ({
      outputTypes: [{ kind: "unknown" }],
      apply: args => {
        if (args.length !== 1) {
          throw new RuntimeError(`${unit.name}: expected 1 argument`);
        }
        const a = args[0];
        if (isDuration(a)) {
          return RTV.num(durationSeconds(a) / unit.factorToSeconds);
        }
        return makeDuration(toNumber(a) * unit.factorToSeconds);
      },
    }),
  });
}

// ── Arithmetic dispatch for datetime / duration ────────────────────────────

/**
 * Attempt to handle a named binary operator ("plus", "minus", "lt", ...)
 * when at least one operand is a datetime or duration class_instance.
 * Returns the result value on success, or undefined to let the generic
 * numeric path run.
 */
export function tryDatetimeDurationBinop(
  opName: string,
  a: RuntimeValue,
  b: RuntimeValue
): RuntimeValue | undefined {
  const aDt = isDatetime(a);
  const bDt = isDatetime(b);
  const aDur = isDuration(a);
  const bDur = isDuration(b);
  if (!aDt && !bDt && !aDur && !bDur) return undefined;

  if (opName === "minus") {
    if (aDt && bDt) {
      const secs = (datetimeToUtcMs(a) - datetimeToUtcMs(b)) / 1000;
      return makeDuration(secs);
    }
    if (aDt && bDur) {
      const ms = datetimeToUtcMs(a) - durationSeconds(b) * 1000;
      return datetimeFromUtcMs(ms);
    }
    if (aDur && bDur) {
      return makeDuration(durationSeconds(a) - durationSeconds(b));
    }
  }
  if (opName === "plus") {
    if (aDt && bDur) {
      const ms = datetimeToUtcMs(a) + durationSeconds(b) * 1000;
      return datetimeFromUtcMs(ms);
    }
    if (aDur && bDt) {
      const ms = datetimeToUtcMs(b) + durationSeconds(a) * 1000;
      return datetimeFromUtcMs(ms);
    }
    if (aDur && bDur) {
      return makeDuration(durationSeconds(a) + durationSeconds(b));
    }
  }
  if (
    opName === "lt" ||
    opName === "le" ||
    opName === "gt" ||
    opName === "ge" ||
    opName === "eq" ||
    opName === "ne"
  ) {
    let lhs: number | undefined;
    let rhs: number | undefined;
    if (aDt && bDt) {
      lhs = datetimeToUtcMs(a);
      rhs = datetimeToUtcMs(b);
    } else if (aDur && bDur) {
      lhs = durationSeconds(a);
      rhs = durationSeconds(b);
    }
    if (lhs !== undefined && rhs !== undefined) {
      switch (opName) {
        case "lt":
          return RTV.logical(lhs < rhs);
        case "le":
          return RTV.logical(lhs <= rhs);
        case "gt":
          return RTV.logical(lhs > rhs);
        case "ge":
          return RTV.logical(lhs >= rhs);
        case "eq":
          return RTV.logical(lhs === rhs);
        case "ne":
          return RTV.logical(lhs !== rhs);
      }
    }
  }
  throw new RuntimeError(
    `Unsupported operator '${opName}' for datetime/duration`
  );
}
