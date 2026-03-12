classdef ConcreteRect_ < AbstractShape_
  properties
    Width
    Height
  end

  methods
    function obj = ConcreteRect_(w, h, c)
      obj.Width = w;
      obj.Height = h;
      obj.Color = c;
    end

    function a = area(obj)
      a = obj.Width * obj.Height;
    end

    function p = perimeter(obj)
      p = 2 * (obj.Width + obj.Height);
    end
  end
end
