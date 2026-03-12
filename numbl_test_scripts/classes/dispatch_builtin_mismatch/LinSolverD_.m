classdef LinSolverD_
  properties
    Data = 0
  end
  methods
    function obj = LinSolverD_(d)
      obj.Data = d;
    end
    function r = linsolve(obj, A, B)
      % Custom 3-arg linsolve (obj + A + B) — returns obj.Data
      % Builtin linsolve takes exactly 2 args (A, B), so 3-arg form
      % can only be this class method.
      r = obj.Data + A(1) + B(1);
    end
  end
end
