// JS sibling of `plot_dispatch.h`. Two delivery modes:
//
// 1. Host hook mode (numbl JIT bridge): when the host binds a
//    `$plotDispatch(name, args)` callback into the emit's `$h`
//    helpers, the helper forwards the raw mtoc2-shaped args to the
//    host. The host is expected to translate the args into its own
//    runtime value shape and route them into its plot pipeline
//    (numbl: `dispatchPlotBuiltin` into `rt.plotInstructions`).
//
//    Wire-format encoding is skipped entirely in this mode — the
//    host operates in-process and consumes the same value graph the
//    emitted code already holds.
//
// 2. Wire-format mode (standalone AOT / viewer tee): when no host
//    hook is present, fall back to the RS-prefixed JSON record on
//    stdout that the launcher splits and forwards. Same shape the C
//    sibling emits, so a single viewer parses both backends.
//
// Wire shape per call (mode 2):
//   \x1e mtoc2:plot \t {"call":"<name>","args":[<arg>, ...]} \n
//
// Arg encoding (per source-level arg):
//   number → bare numeric (non-finite → null, matching the C side
//            and JSON.stringify's natural behavior)
//   string → {"kind":"text","data":"<text>"}
//   char   → same as string (MATLAB plot semantics don't distinguish)
//   tensor → {"kind":"tensor","dims":[…],"data":[…]} (column-major
//            flatten; non-finite as null)

function isTensor(v) {
  return typeof v === "object" && v !== null && v.mtoc2Tag === "tensor";
}
function isChar(v) {
  return typeof v === "object" && v !== null && v.mtoc2Tag === "char";
}

function encodeArg(v) {
  if (typeof v === "number") {
    return Number.isFinite(v) ? v : null;
  }
  if (typeof v === "boolean") return v ? 1 : 0;
  if (typeof v === "string") return { kind: "text", data: v };
  if (isChar(v)) return { kind: "text", data: v.value };
  if (isTensor(v)) {
    const data = new Array(v.data.length);
    for (let i = 0; i < v.data.length; i++) {
      const x = v.data[i];
      data[i] = Number.isFinite(x) ? x : null;
    }
    return { kind: "tensor", dims: v.shape.slice(), data };
  }
  // Unknown shape — pass through. Type-level rejection on the
  // lowering side keeps complex/struct/class/handle out of here.
  return String(v);
}

export function mtoc2_plot_dispatch(name, ...args) {
  if (typeof globalThis.$plotDispatch === "function") {
    globalThis.$plotDispatch(name, args);
    return;
  }
  const record = { call: name, args: args.map(encodeArg) };
  $write("\x1emtoc2:plot\t" + JSON.stringify(record) + "\n");
}
