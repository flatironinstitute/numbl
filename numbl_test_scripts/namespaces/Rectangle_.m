classdef Rectangle_
  properties
    width
    height
  end

  methods
    function obj = Rectangle_(w, h)
      obj.width = w;
      obj.height = h;
    end

    function a = area(obj)
      a = obj.width * obj.height;
    end

    function p = perimeter(obj)
      p = 2 * (obj.width + obj.height);
    end
  end
end
