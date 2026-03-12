classdef MultiplierD_
  properties
    Val = 1
  end
  methods
    function obj = MultiplierD_(v)
      obj.Val = v;
    end
    function r = apply_(obj, x)
      r = obj.Val * x;
    end
  end
end
