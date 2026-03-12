% Define a class with a double_it method
classdef Doubler_
  properties
    Factor = 2
  end
  methods
    function obj = Doubler_(f)
      obj.Factor = f;
    end
    function r = double_it(obj, x)
      r = x * obj.Factor;
    end
  end
end