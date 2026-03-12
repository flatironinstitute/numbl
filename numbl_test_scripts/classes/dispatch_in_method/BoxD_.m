classdef BoxD_
  properties
    W = 0
    H = 0
  end
  methods
    function obj = BoxD_(w, h)
      obj.W = w;
      obj.H = h;
    end
    function r = area_(obj)
      r = obj.W * obj.H;
    end
    function r = double_area_(obj)
      % Calls area_ using function-call syntax inside a method body.
      % In MATLAB, this should call the class method area_(obj)
      % because there is no local function "area_" in this file,
      % and class methods take precedence over file functions.
      r = area_(obj) * 2;
    end
    function r = area_plus_(obj, extra)
      % Mix dot syntax and function-call syntax
      a = obj.area_();
      r = a + extra;
    end
  end
end
