classdef IsPropClass_
  % Helper class for test_isprop.m: a public property, a private property,
  % and a method (so isprop can distinguish properties from methods).
  properties
    width = 10
    height = 20
  end
  properties (Access = private)
    secret = 99
  end
  methods
    function r = area(obj)
      r = obj.width * obj.height;
    end
  end
end
