classdef CopyablePose_ < matlab.mixin.Copyable
  % Helper for handle_copyable_setters.m. Derives from matlab.mixin.Copyable,
  % which is a handle base class, so instances have reference semantics. The
  % setter has no output (handle-style) and converts its input; the getter
  % backs a Dependent property.
  properties
    r = []
  end
  properties (Dependent)
    rsum
  end
  methods
    function set.r(obj, v)
      if isempty(v)
        return;
      end
      assert(length(v) == 3, 'r must be a 3-vector');
      obj.r = v(:).';
    end
    function s = get.rsum(obj)
      s = sum(obj.r);
    end
  end
end
