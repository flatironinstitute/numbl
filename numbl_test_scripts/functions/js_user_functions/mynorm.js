// native: mynorm
// wasm: mynorm
register({
  check: function (argTypes, nargout) {
    return { outputTypes: [IType.num()] };
  },
  apply: function (args, nargout) {
    var x = args[0];
    var data = x.data; // Float64Array

    if (native) {
      // Native path: koffi can pass Float64Array directly as a pointer
      var mynorm_native = native.func("double mynorm(int n, double *x)");
      return RTV.num(mynorm_native(data.length, data));
    }

    if (wasm) {
      // Wasm path: must copy data into wasm linear memory since wasm
      // cannot directly access JS typed arrays
      var n = data.length;
      var ptr = wasm.exports.alloc_doubles(n);
      var mem = new Float64Array(wasm.exports.memory.buffer, ptr, n);
      mem.set(data);
      var result = wasm.exports.mynorm(n, ptr);
      wasm.exports.free_doubles();
      return RTV.num(result);
    }

    // JS fallback
    var sum = 0;
    for (var i = 0; i < data.length; i++) {
      sum += data[i] * data[i];
    }
    return RTV.num(Math.sqrt(sum));
  },
});
