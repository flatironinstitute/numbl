classdef ArgCtor
% Mirrors surfaceop's constructor pattern: an arguments block whose default
% for a later positional argument calls a static method on an earlier
% argument, plus a name-value struct with defaults. The output variable is
% prepended to the parameter list, so argument positions are offset by one.
  properties
    a
    mi
    method
  end
  methods
    function obj = ArgCtor(a, mi, opts)
      arguments
        a = 0
        mi = ArgCtor.defaultMi(a)
        opts.method = 'DtN'
      end
      obj.a = a;
      obj.mi = mi;
      obj.method = opts.method;
    end
  end
  methods ( Static )
    function out = defaultMi(a)
      out = a + 1000;
    end
  end
end
