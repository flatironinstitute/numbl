classdef EndSizeOverload_
% Helper for test_end_class_arrays: overloads size and numel but not end.
% MATLAB's builtin end uses the builtin dimensions, not these overloads.
  properties
    v
  end
  methods
    function obj = EndSizeOverload_(v)
      if nargin > 0
        obj.v = v;
      end
    end
    function s = size(obj, varargin)
      s = [5 7];
    end
    function n = numel(obj)
      n = 35;
    end
  end
end
