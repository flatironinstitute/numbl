// wasm: wadd
register({
  resolve: function (argTypes, nargout) {
    return {
      outputTypes: [{ kind: "number" }],
      apply: function (args) {
        if (wasm) {
          return wasm.exports.wadd(args[0], args[1]);
        }
        return args[0] + args[1];
      },
    };
  },
});
