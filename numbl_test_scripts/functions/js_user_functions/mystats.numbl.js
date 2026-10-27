register({
  help:
    "MYSTATS  Mean and range of a vector.\n" +
    "   [m, r] = mystats(x) returns the mean m and the range r of x.",
  apply: function (args, nargout) {
    var x = args[0].data;
    var sum = 0;
    var lo = Infinity;
    var hi = -Infinity;
    for (var i = 0; i < x.length; i++) {
      sum += x[i];
      if (x[i] < lo) lo = x[i];
      if (x[i] > hi) hi = x[i];
    }
    var m = sum / x.length;
    if (nargout <= 1) return m;
    return [m, hi - lo];
  },
});
