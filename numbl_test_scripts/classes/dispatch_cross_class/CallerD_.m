classdef CallerD_
  % CallerD_ is lowered before LinSolverD_ (alphabetical order).
  % Its method calls linsolve(obj, A, B) where obj could be a LinSolverD_.
  % At lowering time LinSolverD_ isn't lowered yet, so classSignatures
  % (populated from the AST) must be used to know linsolve is a class method.
  methods
    function r = run(~, obj, A, B)
      r = linsolve(obj, A, B);
    end
  end
end
