// A function with no outputs: returning undefined means "nothing".
register(function (args) {
  if (args.length !== 1) throw new RuntimeError("mynoout expects 1 argument");
});
