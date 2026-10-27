// The original register({ resolve }) form, kept to test that it still works.
register({
  resolve: function (argTypes, nargout) {
    return {
      outputTypes: [{ kind: "number" }],
      apply: function (args) {
        return args[0] * args[1];
      },
    };
  },
});
