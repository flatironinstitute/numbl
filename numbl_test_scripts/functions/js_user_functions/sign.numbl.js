// Override the builtin sign to verify .js beats builtins
register(function (args) {
  return 999;
});
