% Test: cross-class dispatch where the target class isn't lowered yet.
%
% CallerD_ (lowered first, alphabetically) has a method that calls
% linsolve(obj, A, B). linsolve is a builtin (2 args) and a method of
% LinSolverD_ (3 args). When CallerD_ is lowered, LinSolverD_ hasn't
% been lowered yet, so the compiler must check classSignatures (AST)
% to know that linsolve is a class method and not error out.

solver = LinSolverD_(100);
caller = CallerD_();

r = caller.run(solver, [1 2], [3 4]);
assert(r == 104);  % 100 + 1 + 3

disp('SUCCESS')
