classdef OpVal
  % A value class that overloads the unary operators and a few binary ones.
  properties
    v
    tag = ''
  end
  methods
    function o = OpVal(v, tag)
      o.v = v;
      if nargin > 1
        o.tag = tag;
      end
    end
    function r = not(a)
      r = OpVal(~a.v, 'not');
    end
    function r = uminus(a)
      r = OpVal(-a.v, 'uminus');
    end
    function r = uplus(a)
      r = OpVal(a.v, 'uplus');
    end
    function r = ctranspose(a)
      r = OpVal(a.v', 'ctranspose');
    end
    function r = transpose(a)
      r = OpVal(a.v.', 'transpose');
    end
    function r = and(a, b)
      r = OpVal(val(a) & val(b), 'and');
    end
    function r = or(a, b)
      r = OpVal(val(a) | val(b), 'or');
    end
    function r = plus(a, b)
      r = OpVal(val(a) + val(b), 'plus');
    end
  end
end

function x = val(a)
  if isa(a, 'OpVal')
    x = [a.v];
  else
    x = a;
  end
end
