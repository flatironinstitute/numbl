classdef IdxObj_
  % Helper for test_object_array_indexing.m: a plain value class.
  properties
    x = 0;
  end
  methods
    function obj = IdxObj_(v)
      if nargin > 0
        obj.x = v;
      end
    end
  end
end
