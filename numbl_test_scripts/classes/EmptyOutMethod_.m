classdef EmptyOutMethod_ < handle
  % Exercises a class method declared with an explicit empty output list
  % `function [] = name(...)`, and a `~` placeholder param in an
  % anonymous function used inside a method.
  properties
    Value = 0;
  end
  methods
    function [] = doThing(obj, x)
      adder = @(~, v) v + 1;
      obj.Value = adder(99, x);
    end
  end
end
