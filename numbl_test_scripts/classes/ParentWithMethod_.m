classdef ParentWithMethod_
  properties
    Val
  end
  methods
    function obj = ParentWithMethod_(v)
      obj.Val = v;
    end
    function r = greet(obj)
      r = obj.Val;
    end
  end
end
