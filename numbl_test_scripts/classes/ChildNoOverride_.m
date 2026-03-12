classdef ChildNoOverride_ < ParentWithMethod_
  properties
    Tag
  end
  methods
    function obj = ChildNoOverride_(v, tag)
      obj = obj@ParentWithMethod_(v);
      obj.Tag = tag;
    end
    function r = childOnly(obj)
      r = obj.Tag;
    end
  end
end
