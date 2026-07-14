/**
 * Formatter for datetime `Format` patterns — the Unicode-LDML-style tokens
 * MATLAB uses (e.g. "yyyy-MM-dd'T'HH:mm:ss'Z'").
 *
 * Supported tokens: y/u (year), M (month; MMM abbreviated name, MMMM full
 * name), d (day of month), H (hour 0–23), h (hour 1–12), m (minute),
 * s (second), S (fractional-second digits), a (AM/PM). Text between single
 * quotes is emitted literally; '' is an escaped single quote.
 */

import { RuntimeError } from "./error.js";

const MONTH_ABBR = [
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

const MONTH_FULL = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
];

function pad(n: number, width: number): string {
  return String(n).padStart(width, "0");
}

export function formatDatetimePattern(
  pattern: string,
  year: number,
  month: number,
  day: number,
  hour: number,
  minute: number,
  second: number
): string {
  const wholeSec = Math.floor(second);
  const frac = second - wholeSec;
  const monthIdx = Math.max(0, Math.min(11, Math.floor(month) - 1));

  const formatToken = (c: string, n: number): string => {
    switch (c) {
      case "y":
      case "u":
        if (n === 2) return pad(Math.floor(year) % 100, 2);
        return pad(Math.floor(year), n);
      case "M":
        if (n === 1) return String(Math.floor(month));
        if (n === 2) return pad(Math.floor(month), 2);
        if (n === 3) return MONTH_ABBR[monthIdx];
        return MONTH_FULL[monthIdx];
      case "d":
        return pad(Math.floor(day), n);
      case "H":
        return pad(Math.floor(hour), n);
      case "h":
        return pad(Math.floor(hour) % 12 || 12, n);
      case "m":
        return pad(Math.floor(minute), n);
      case "s":
        return pad(wholeSec, n);
      case "S":
        return pad(Math.floor(frac * Math.pow(10, n)), n);
      case "a":
        return hour < 12 ? "AM" : "PM";
      default:
        throw new RuntimeError(
          `datetime: unsupported format token '${c.repeat(n)}' in '${pattern}'`
        );
    }
  };

  let out = "";
  let i = 0;
  while (i < pattern.length) {
    const c = pattern[i];
    if (c === "'") {
      // Quoted literal; '' inside (or standing alone) is an escaped quote.
      i++;
      let lit = "";
      let closed = false;
      while (i < pattern.length) {
        if (pattern[i] === "'") {
          if (pattern[i + 1] === "'") {
            lit += "'";
            i += 2;
            continue;
          }
          i++;
          closed = true;
          break;
        }
        lit += pattern[i++];
      }
      if (!closed && lit !== "")
        throw new RuntimeError(
          `datetime: unterminated quoted literal in format '${pattern}'`
        );
      out += lit === "" ? "'" : lit;
      continue;
    }
    if (!/[a-zA-Z]/.test(c)) {
      out += c;
      i++;
      continue;
    }
    let j = i;
    while (j < pattern.length && pattern[j] === c) j++;
    out += formatToken(c, j - i);
    i = j;
  }
  return out;
}
