// wasm: wadd
register({
  check: function (argTypes, nargout) {
    return { outputTypes: [IType.num()] };
  },
  apply: function (args, nargout) {
    if (wasm) {
      return RTV.num(wasm.exports.wadd(args[0], args[1]));
    }
    return RTV.num(args[0] + args[1]);
  },
});
