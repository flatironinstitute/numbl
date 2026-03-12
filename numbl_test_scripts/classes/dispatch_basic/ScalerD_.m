classdef ScalerD_
  properties
    Factor = 1
  end
  methods
    function obj = ScalerD_(f)
      obj.Factor = f;
    end
    function r = transform(obj, x)
      r = x * obj.Factor;
    end
  end
end
