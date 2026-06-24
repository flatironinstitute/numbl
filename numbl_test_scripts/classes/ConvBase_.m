classdef ConvBase_
  % Helper for constructor_name_collision.m. Defines a converter method named
  % ConvLeaf_ (same name as the subclass), which ConvLeaf_ inherits. This makes
  % the subclass have an instance method whose name equals the subclass's own
  % constructor name — the situation that must still resolve `ConvLeaf_(x)` to
  % construction, not method dispatch.
  properties
    data = 0
  end
  methods
    function r = ConvLeaf_(obj)
      r = ConvLeaf_();
      r.data = obj.data;
    end
  end
end
