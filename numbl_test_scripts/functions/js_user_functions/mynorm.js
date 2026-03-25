// native: mynorm
// wasm: mynorm
register({
  resolve: function (argTypes, nargout) {
    return {
      outputTypes: [{ kind: "number" }],
      apply: function (args) {
        var x = args[0];
        var data = x.data; // Float64Array

        if (native) {
          var mynorm_native = native.func("double mynorm(int n, double *x)");
          return RTV.num(mynorm_native(data.length, data));
        }

        if (wasm) {
          var n = data.length;
          var ptr = wasm.exports.alloc_doubles(n);
          var mem = new Float64Array(wasm.exports.memory.buffer, ptr, n);
          mem.set(data);
          var result = wasm.exports.mynorm(n, ptr);
          wasm.exports.free_doubles();
          return result;
        }

        // JS fallback
        var sum = 0;
        for (var i = 0; i < data.length; i++) {
          sum += data[i] * data[i];
        }
        return Math.sqrt(sum);
      },
    };
  },
});
