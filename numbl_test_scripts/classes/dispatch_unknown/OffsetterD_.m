classdef OffsetterD_
  properties
    Offset = 0
  end
  methods
    function obj = OffsetterD_(o)
      obj.Offset = o;
    end
    function r = calc(obj, x)
      r = x + obj.Offset;
    end
  end
end
