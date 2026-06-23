classdef RBox
  properties
    v = 0
  end
  methods
    function obj = RBox(x)
      if nargin > 0, obj.v = x; end
    end
  end
end
