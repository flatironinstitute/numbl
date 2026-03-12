% Test: builtin arg-count mismatch should fall through to class method
%
% linsolve builtin: linsolve(A, B) — exactly 2 args, 1 output
% LinSolverD_ class: linsolve(obj, A, B) — 3 args in function-call syntax
%
% When the first arg type is unknown and the arg count doesn't match
% the builtin, the compiler should assume it's a class method call
% (not throw a compile error).

% Test 1: dot syntax — always calls class method
s = LinSolverD_(100);
r1 = s.linsolve([1 2], [3 4]);
assert(r1 == 104);  % 100 + 1 + 3

% Test 2: unknown type, 3 args — can't be builtin, must be class method
X = 0;
X = LinSolverD_(50);
r2 = linsolve(X, [10 20], [30 40]);
assert(r2 == 90);  % 50 + 10 + 30

disp('SUCCESS')
