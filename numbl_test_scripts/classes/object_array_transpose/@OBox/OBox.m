classdef OBox
  properties
    v = 0
  end
  methods
    function obj = OBox(x)
      if nargin > 0, obj.v = x; end
    end
  end
end
