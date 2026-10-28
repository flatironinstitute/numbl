/**
 * Web builtins on the runtime's FileIOAdapter: websave, webread.
 */

import {
  type RuntimeValue,
  RTV,
  toString,
  toNumber,
  RuntimeError,
} from "../../runtime/index.js";
import { isRuntimeStruct } from "../../runtime/types.js";
import type { WebOptions } from "../../fileIOAdapter.js";
import { convertJsonValue } from "./misc.js";
import { requireFileIO } from "./filesystem.js";
import { builtin } from "./builtin.js";

/** Extract WebOptions from a weboptions struct, or return undefined. */
function extractWebOptions(arg: RuntimeValue): WebOptions | undefined {
  if (!isRuntimeStruct(arg)) return undefined;
  const s = arg;
  // Detect weboptions struct by checking for the Timeout field
  if (!s.fields.has("Timeout")) return undefined;
  const opts: WebOptions = {};
  const t = s.fields.get("Timeout");
  if (t !== undefined) opts.timeout = toNumber(t);
  const rm = s.fields.get("RequestMethod");
  if (rm !== undefined) {
    const v = toString(rm);
    if (v !== "auto") opts.requestMethod = v;
  }
  const un = s.fields.get("Username");
  if (un !== undefined) {
    const v = toString(un);
    if (v) opts.username = v;
  }
  const pw = s.fields.get("Password");
  if (pw !== undefined) {
    const v = toString(pw);
    if (v) opts.password = v;
  }
  const kn = s.fields.get("KeyName");
  if (kn !== undefined) {
    const v = toString(kn);
    if (v) opts.keyName = v;
  }
  const kv = s.fields.get("KeyValue");
  if (kv !== undefined) {
    const v = toString(kv);
    if (v) opts.keyValue = v;
  }
  return opts;
}

/** Append the name-value pairs in args[first..] (stopping before a
 *  trailing weboptions struct) to `url` as query parameters. Returns the
 *  URL and the options. */
function buildRequest(
  url: string,
  args: RuntimeValue[],
  first: number
): { url: string; webOpts: WebOptions | undefined } {
  let webOpts: WebOptions | undefined;
  let queryEnd = args.length;
  if (args.length > first) {
    webOpts = extractWebOptions(args[args.length - 1]);
    if (webOpts) queryEnd = args.length - 1;
  }
  const queryParts: string[] = [];
  for (let i = first; i + 1 < queryEnd; i += 2) {
    const name = encodeURIComponent(toString(args[i]));
    const value = encodeURIComponent(toString(args[i + 1]));
    queryParts.push(`${name}=${value}`);
  }
  if (queryParts.length > 0) {
    const sep = url.includes("?") ? "&" : "?";
    url += sep + queryParts.join("&");
  }
  return { url, webOpts };
}

builtin({
  name: "websave",
  nargin: [2, Infinity],
  mfile: true,
  apply: (args, _nargout, ctx) => {
    const io = requireFileIO(ctx.rt);
    if (!io.websave)
      throw new RuntimeError("websave is not available in this environment");
    const filename = toString(args[0]);
    const { url, webOpts } = buildRequest(toString(args[1]), args, 2);
    io.websave(url, filename, webOpts);
    return RTV.char(filename);
  },
});

builtin({
  name: "webread",
  nargin: [1, Infinity],
  mfile: true,
  apply: (args, _nargout, ctx) => {
    const io = requireFileIO(ctx.rt);
    if (!io.webread)
      throw new RuntimeError("webread is not available in this environment");
    const { url, webOpts } = buildRequest(toString(args[0]), args, 1);
    const text = io.webread(url, webOpts);
    // Try to parse as JSON (MATLAB auto-decodes JSON responses)
    try {
      const parsed = JSON.parse(text);
      return convertJsonValue(parsed);
    } catch {
      // Not JSON — return as char array
      return RTV.char(text);
    }
  },
});
