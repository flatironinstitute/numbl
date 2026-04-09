var H = importJS("_test_helpers");

register({
  resolve: function (argTypes, nargout) {
    return {
      outputTypes: [{ kind: "number" }],
      apply: function (args) {
        return H.add3(args[0], args[1], args[2]);
      },
    };
  },
});
