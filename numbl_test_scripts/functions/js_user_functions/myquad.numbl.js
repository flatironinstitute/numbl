// wasm: myquad
// Numerically integrate a function handle over [a, b] via the midpoint rule.
//   myquad(@(x) x.^2, 0, 1)  ->  ~1/3
// Demonstrates passing a numbl function handle through to WASM, which calls
// it back through the host `numbl_cb_d` import. Falls back to a pure-JS loop
// (still invoking the handle via callHandle) when the WASM module is absent.
register({
  resolve: function (argTypes, nargout) {
    return {
      outputTypes: [{ kind: "number" }],
      apply: function (args) {
        var f = args[0]; // RuntimeFunction handle
        var a = toNumber(args[1]);
        var b = toNumber(args[2]);
        var n = 1000;

        if (wasm) {
          // Expose the handle to WASM as a scalar->scalar callback id.
          var id = wasm.callbacks.add(function (x) {
            return toNumber(callHandle(f, [x]));
          });
          try {
            return wasm.exports.myquad(id, a, b, n);
          } finally {
            wasm.callbacks.remove(id);
          }
        }

        // JS fallback: same algorithm, invoking the handle directly.
        var h = (b - a) / n;
        var sum = 0;
        for (var i = 0; i < n; i++) {
          sum += toNumber(callHandle(f, [a + (i + 0.5) * h]));
        }
        return sum * h;
      },
    };
  },
});
