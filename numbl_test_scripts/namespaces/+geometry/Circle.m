classdef Circle
  properties
    radius
  end

  methods
    function obj = Circle(r)
      obj.radius = r;
    end

    function a = area(obj)
      a = pi * obj.radius^2;
    end

    function c = circumference(obj)
      c = 2 * pi * obj.radius;
    end
  end

  methods (Static)
    function r = unit_radius()
      r = 1;
    end
  end
end
