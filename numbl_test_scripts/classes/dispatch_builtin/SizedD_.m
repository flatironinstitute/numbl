classdef SizedD_
  properties
    Data = 0
  end
  methods
    function obj = SizedD_(d)
      obj.Data = d;
    end
    function r = size(obj)
      % Custom size method that returns -42 (to distinguish from builtin)
      r = -42;
    end
    function r = length(obj)
      % Custom length method
      r = -99;
    end
  end
end
