classdef ShapeD_
  properties
    W = 0
    H = 0
  end
  methods
    function obj = ShapeD_(w, h)
      obj.W = w;
      obj.H = h;
    end
    function r = area_(obj)
      r = obj.W * obj.H;
    end
  end
end
