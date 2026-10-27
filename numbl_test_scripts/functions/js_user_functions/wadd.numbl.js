// wasm: wadd
register(function (args) {
  if (wasm) {
    return wasm.exports.wadd(args[0], args[1]);
  }
  return args[0] + args[1];
});
