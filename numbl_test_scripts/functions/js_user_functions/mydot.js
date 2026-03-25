// native: mydot
register({
  resolve: function (argTypes, nargout) {
    return {
      outputTypes: [{ kind: "number" }],
      apply: function (args) {
        var a = args[0];
        var b = args[1];
        if (native) {
          var mydot_native = native.func(
            "double mydot(int n, double *a, double *b)"
          );
          return RTV.num(mydot_native(a.data.length, a.data, b.data));
        }
        // JS fallback
        var sum = 0;
        for (var i = 0; i < a.data.length; i++) {
          sum += a.data[i] * b.data[i];
        }
        return sum;
      },
    };
  },
});
