% Test xor on arrays and various types

% Element-wise xor on numeric arrays
a = [1 0 1 0];
b = [1 1 0 0];
r = xor(a, b);
assert(all(r == [0 1 1 0]));

% xor on logical arrays
la = logical([1 0 1 0]);
lb = logical([1 1 0 0]);
assert(all(xor(la, lb) == [0 1 1 0]));

% xor on matrices
A = [1 0; 0 1];
B = [1 1; 0 0];
R = xor(A, B);
assert(all(all(R == [0 1; 0 1])));

% xor with scalar and array
assert(all(xor(1, [0 1 0 1]) == [1 0 1 0]));
assert(all(xor([0 1 0 1], 0) == [0 1 0 1]));

% Scalar xor still works
assert(xor(true, false) == true);
assert(xor(true, true) == false);
assert(xor(false, false) == false);

disp('SUCCESS');
