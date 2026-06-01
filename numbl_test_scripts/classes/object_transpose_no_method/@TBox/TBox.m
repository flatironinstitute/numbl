classdef TBox
  properties
    v = 0
  end
  methods
    function obj = TBox(x)
      obj.v = x;
    end
    function r = getv(obj)
      r = obj.v;
    end
  end
end
