classdef TriArea_
  properties
    Base = 1
    Height = 1
  end
  properties (Dependent)
    Area
  end
  methods
    function a = get.Area(obj)
      a = 0.5 * obj.Base * obj.Height;
    end
  end
end
