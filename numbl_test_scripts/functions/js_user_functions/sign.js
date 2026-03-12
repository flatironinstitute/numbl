// Override the builtin sign to verify .js beats builtins
register({
  apply: function (args, nargout) {
    return RTV.num(999);
  },
});
