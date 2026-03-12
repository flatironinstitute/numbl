classdef AdderD_
  properties
    Val = 0
  end
  methods
    function obj = AdderD_(v)
      obj.Val = v;
    end
    function r = apply_(obj, x)
      r = obj.Val + x;
    end
  end
end
