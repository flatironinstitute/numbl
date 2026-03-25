// Override the builtin sign to verify .js beats builtins
register({
  resolve: function (argTypes, nargout) {
    return {
      outputTypes: [{ kind: "number" }],
      apply: function (args) {
        return 999;
      },
    };
  },
});
