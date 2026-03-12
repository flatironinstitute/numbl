register({
  check: function (argTypes, nargout) {
    return { outputTypes: [IType.num()] };
  },
  apply: function (args, nargout) {
    return RTV.num(args[0] + args[1]);
  },
});
