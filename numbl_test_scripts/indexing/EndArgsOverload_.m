classdef EndArgsOverload_
% Helper for test_end_class_arrays: overloaded end returns the number of
% subscripts n, so tests can check that the overload is called (also for
% object arrays) and receives n.
  properties
    v
  end
  methods
    function obj = EndArgsOverload_(v)
      if nargin > 0
        obj.v = v;
      end
    end
    function e = end(obj, k, n)
      e = n;
    end
  end
end
