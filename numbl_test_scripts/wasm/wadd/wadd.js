register({
  check: function (argTypes, nargout) {
    return { outputTypes: [IType.num()] };
  },
  apply: function (args, nargout) {
    return RTV.num(wasm.exports.wadd(args[0], args[1]));
  },
});
