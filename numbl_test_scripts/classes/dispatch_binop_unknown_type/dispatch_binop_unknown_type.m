% Test that binary operations between Unknown-typed class instances and
% known scalar/tensor types preserve Unknown result type, so subsequent
% function calls dispatch to class methods instead of builtins.
%
% Bug: the type inference for Binary expressions incorrectly infers a
% concrete type (ComplexNumber, Tensor) when one operand is Unknown,
% causing downstream function calls (like norm) to call the builtin
% instead of the class method.

% Create via helper so the return type is Unknown at compile time.
obj = make_opclass(10);

% ── Case 1: Unknown minus ComplexNumber ──
% obj - (1+2i) should call @OpClass/minus, returning an OpClass.
% Then norm() should dispatch to @OpClass/norm, not the builtin.
c = 1 + 2i;
result1 = obj - c;
assert(isa(result1, 'OpClass'), 'obj - complex should return OpClass');
n1 = norm(result1);
assert(abs(n1 - abs(10 - (1+2i))) < 1e-10, 'norm(obj - complex) should use class norm');

% ── Case 2: ComplexNumber minus Unknown ──
% (1+2i) - obj should also call @OpClass/minus (MATLAB checks all args).
result2 = c - obj;
assert(isa(result2, 'OpClass'), 'complex - obj should return OpClass');
n2 = norm(result2);
assert(abs(n2 - abs((1+2i) - 10)) < 1e-10, 'norm(complex - obj) should use class norm');

% ── Case 3: Unknown plus Number ──
% obj + 5 should call @OpClass/plus.
% Then sum() should dispatch to @OpClass/sum.
result3 = obj + 5;
assert(isa(result3, 'OpClass'), 'obj + number should return OpClass');
s3 = sum(result3);
assert(s3 == 15, 'sum(obj + number) should use class sum');

% ── Case 4: Unknown minus scalar from sqrt ──
% sqrt(N) returns ComplexNumber type in the compiler.
% obj - sqrt(4) should still dispatch to class minus.
result4 = obj - sqrt(4);
assert(isa(result4, 'OpClass'), 'obj - sqrt(N) should return OpClass');
n4 = norm(result4);
assert(abs(n4 - 8) < 1e-10, 'norm(obj - sqrt(N)) should use class norm');

% ── Case 5: Chained operations preserving class type ──
% (obj - complex) + number should still be an OpClass.
result5 = (obj - c) + 3;
assert(isa(result5, 'OpClass'), 'chained ops should preserve OpClass type');
n5 = norm(result5);
assert(abs(n5 - abs(10 - (1+2i) + 3)) < 1e-10, 'norm of chained ops should use class norm');

% ── Case 6: norm with two args (class instance result + number) ──
% norm(obj - c, 2) should dispatch to @OpClass/norm.
result6 = obj - c;
n6 = norm(result6, 2);
assert(abs(n6 - abs(10 - (1+2i))) < 1e-10, 'norm(obj-complex, 2) should use class norm');

% ── Case 7: abs on class result of Unknown op ComplexNumber ──
% abs(obj - c) should dispatch to @OpClass/abs.
result7 = abs(obj - c);
assert(isa(result7, 'OpClass'), 'abs(obj - complex) should return OpClass');

disp('SUCCESS');
