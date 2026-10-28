classdef PropValidHandle_ < handle
  properties
    a (1,1) double {mustBePositive} = 1
  end
  methods
    function obj = PropValidHandle_(v)
      if nargin > 0
        obj.a = v;
      end
    end
  end
end
