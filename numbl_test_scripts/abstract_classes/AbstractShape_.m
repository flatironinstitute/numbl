classdef (Abstract) AbstractShape_
  properties
    Color
  end

  methods (Abstract)
    a = area(obj)
    p = perimeter(obj)
  end

  methods
    function desc = describe(obj)
      desc = obj.Color;
    end
  end
end
