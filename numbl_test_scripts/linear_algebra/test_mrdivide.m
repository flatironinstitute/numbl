% Test matrix right division (mrdivide, /)
% A / B should equal A * inv(B), not element-wise division

A = [1 2; 3 4];
B = [5 6; 7 8];

% Matrix right division
C = A / B;
expected = A * inv(B);
assert(abs(C(1,1) - expected(1,1)) < 1e-10, 'mrdivide (1,1)');
assert(abs(C(1,2) - expected(1,2)) < 1e-10, 'mrdivide (1,2)');
assert(abs(C(2,1) - expected(2,1)) < 1e-10, 'mrdivide (2,1)');
assert(abs(C(2,2) - expected(2,2)) < 1e-10, 'mrdivide (2,2)');

% Scalar division should still work
assert(6/3 == 2, 'scalar div');

% Vector / scalar
v = [6 9 12] / 3;
assert(isequal(v, [2 3 4]), 'vector / scalar');

disp('SUCCESS');
