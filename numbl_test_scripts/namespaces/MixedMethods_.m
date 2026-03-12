classdef MixedMethods_
  methods (Static = true)
    y = func1(x)
    function y = func2(x)
      y = x + 1;
    end
  end
end
