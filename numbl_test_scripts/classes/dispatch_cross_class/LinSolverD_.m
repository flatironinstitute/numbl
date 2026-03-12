classdef LinSolverD_
  properties
    Data = 0
  end
  methods
    function obj = LinSolverD_(d)
      obj.Data = d;
    end
    function r = linsolve(obj, A, B)
      r = obj.Data + A(1) + B(1);
    end
  end
end
