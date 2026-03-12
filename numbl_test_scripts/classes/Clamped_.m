classdef Clamped_
  properties
    Value = 0
  end
  methods
    function obj = set.Value(obj, val)
      if val > 100
        obj.Value = 100;
      else
        obj.Value = val;
      end
    end
  end
end
