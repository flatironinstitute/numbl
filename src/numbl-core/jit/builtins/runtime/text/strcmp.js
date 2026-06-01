/* JS sibling of strcmp.h — input shape mirrors what mtoc2's js-aot /
 * interpreter pass to the helper: either a raw JS string (numbl-style
 * scalar handle) or a `{mtoc2Tag:"char", value:string}` char wrapper.
 * The helper normalizes either form to its raw bytes before comparing.
 */

function _mtoc2_strcmp_text(v) {
  if (typeof v === "string") return v;
  if (v && v.mtoc2Tag === "char" && typeof v.value === "string") return v.value;
  return null;
}

export function mtoc2_strcmp(a, b) {
  const sa = _mtoc2_strcmp_text(a);
  const sb = _mtoc2_strcmp_text(b);
  if (sa === null || sb === null) return 0;
  return sa === sb ? 1 : 0;
}

export function mtoc2_strcmpi(a, b) {
  const sa = _mtoc2_strcmp_text(a);
  const sb = _mtoc2_strcmp_text(b);
  if (sa === null || sb === null) return 0;
  return sa.toLowerCase() === sb.toLowerCase() ? 1 : 0;
}
