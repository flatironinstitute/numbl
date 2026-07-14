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

import type { RuntimeValue } from "../../runtime/types.js";
import {
  RuntimeClassInstance,
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
  return new RuntimeClassInstance("datetime", fields, false);
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

/** Return a copy of a datetime with TimeZone/Format properties attached
 *  (no-op when neither is requested). */
function withTimeZoneFormat(
  dt: RuntimeClassInstance,
  timeZone: string | undefined,
  format: string | undefined
): RuntimeClassInstance {
  if (timeZone === undefined && format === undefined) return dt;
  const fields = new Map(dt.fields);
  if (timeZone !== undefined) fields.set("TimeZone", RTV.char(timeZone));
  if (format !== undefined) fields.set("Format", RTV.char(format));
  return new RuntimeClassInstance("datetime", fields, false);
}

/** Calendar parts of (today + offsetDays), in local time or UTC. */
function todayParts(
  offsetDays: number,
  utc: boolean
): [number, number, number] {
  const now = new Date();
  if (utc) {
    const d = new Date(
      Date.UTC(
        now.getUTCFullYear(),
        now.getUTCMonth(),
        now.getUTCDate() + offsetDays
      )
    );
    return [d.getUTCFullYear(), d.getUTCMonth() + 1, d.getUTCDate()];
  }
  const d = new Date(
    now.getFullYear(),
    now.getMonth(),
    now.getDate() + offsetDays
  );
  return [d.getFullYear(), d.getMonth() + 1, d.getDate()];
}

// ── duration construction / introspection ─────────────────────────────────

export function makeDuration(totalSeconds: number): RuntimeClassInstance {
  const fields = new Map<string, RuntimeValue>([
    ["Seconds", RTV.num(totalSeconds)],
  ]);
  return new RuntimeClassInstance("duration", fields, false);
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
      "t = datetime(..., 'TimeZone', tz, 'Format', fmt)",
    ],
    description:
      "Create a datetime value. Supported dateType values for ConvertFrom: " +
      "'datenum', 'posixtime', 'excel', 'excel1904'. Trailing 'TimeZone' " +
      "(only '', 'local', 'UTC') and 'Format' name-value pairs are " +
      "supported; the Format pattern is used by char() and display.",
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
      // Split off trailing 'TimeZone'/'Format' name-value pairs (any order).
      let timeZone: string | undefined;
      let format: string | undefined;
      let positional = args;
      while (positional.length >= 2) {
        const name = argToString(positional[positional.length - 2]);
        if (name === null) break;
        const lc = name.toLowerCase();
        if (lc !== "timezone" && lc !== "format") break;
        const value = argToString(positional[positional.length - 1]);
        if (value === null)
          throw new RuntimeError(`datetime: ${name} value must be text`);
        if (lc === "timezone") timeZone = value;
        else format = value;
        positional = positional.slice(0, -2);
      }
      const tzLc = timeZone?.toLowerCase();
      if (
        tzLc !== undefined &&
        tzLc !== "" &&
        tzLc !== "local" &&
        tzLc !== "utc"
      ) {
        throw new RuntimeError(
          `datetime: unsupported TimeZone '${timeZone}' (supported: '', 'local', 'UTC')`
        );
      }
      // With TimeZone 'UTC', current-time forms use UTC calendar components.
      const utc = tzLc === "utc";
      const finish = (dt: RuntimeClassInstance) =>
        withTimeZoneFormat(dt, timeZone, format);

      if (positional.length === 0) {
        return finish(
          utc ? datetimeFromUtcMs(Date.now()) : datetimeFromDate(new Date())
        );
      }

      // datetime('now' | 'today' | 'yesterday' | 'tomorrow')
      if (positional.length === 1) {
        const s = argToString(positional[0]);
        if (s !== null) {
          const lc = s.toLowerCase();
          if (lc === "now") {
            return finish(
              utc ? datetimeFromUtcMs(Date.now()) : datetimeFromDate(new Date())
            );
          }
          if (lc === "today" || lc === "yesterday" || lc === "tomorrow") {
            const offset = lc === "tomorrow" ? 1 : lc === "yesterday" ? -1 : 0;
            const [Y, M, D] = todayParts(offset, utc);
            return finish(makeDatetime(Y, M, D, 0, 0, 0));
          }
          throw new RuntimeError(
            `datetime: unrecognized relative-day string '${s}'`
          );
        }
      }

      // datetime(X, 'ConvertFrom', dateType)
      if (positional.length === 3) {
        const flag = argToString(positional[1]);
        const kind = argToString(positional[2]);
        if (flag !== null && flag.toLowerCase() === "convertfrom") {
          if (kind === null)
            throw new RuntimeError("datetime: dateType must be text");
          const x = toNumber(positional[0]);
          const k = kind.toLowerCase();
          if (k === "datenum") return finish(datetimeFromDatenum(x));
          if (k === "posixtime") return finish(datetimeFromUtcMs(x * 1000));
          if (k === "excel") {
            return finish(datetimeFromUtcMs((x - 25569) * 86400000));
          }
          if (k === "excel1904") {
            return finish(datetimeFromUtcMs((x - 24107) * 86400000));
          }
          throw new RuntimeError(
            `datetime: unsupported ConvertFrom type '${kind}'`
          );
        }
      }

      // datetime(Y, M, D [, H, MI, S [, MS]])
      if (
        positional.length === 3 ||
        positional.length === 6 ||
        positional.length === 7
      ) {
        const Y = toNumber(positional[0]);
        const M = toNumber(positional[1]);
        const D = toNumber(positional[2]);
        const H = positional.length >= 6 ? toNumber(positional[3]) : 0;
        const MI = positional.length >= 6 ? toNumber(positional[4]) : 0;
        const S = positional.length >= 6 ? toNumber(positional[5]) : 0;
        const MS = positional.length >= 7 ? toNumber(positional[6]) : 0;
        return finish(makeDatetime(Y, M, D, H, MI, S + MS / 1000));
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
