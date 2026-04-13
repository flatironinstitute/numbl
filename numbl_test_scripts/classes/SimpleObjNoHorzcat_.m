classdef SimpleObjNoHorzcat_
  properties
    val
  end
  methods
    function obj = SimpleObjNoHorzcat_(v)
      if nargin > 0
        obj.val = v;
      end
    end
  end
end
