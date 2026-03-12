% Test toeplitz builtin

% Test 1: symmetric toeplitz from real vector
T = toeplitz([1 2 3]);
assert(T(1,1) == 1);
assert(T(1,2) == 2);
assert(T(1,3) == 3);
assert(T(2,1) == 2);
assert(T(2,2) == 1);
assert(T(2,3) == 2);
assert(T(3,1) == 3);
assert(T(3,2) == 2);
assert(T(3,3) == 1);

% Test 2: nonsymmetric toeplitz with col and row vectors
% toeplitz([1;2;3], [1 4 5]) -> [1 4 5; 2 1 4; 3 2 1]
T = toeplitz([1;2;3], [1 4 5]);
assert(T(1,1) == 1); assert(T(1,2) == 4); assert(T(1,3) == 5);
assert(T(2,1) == 2); assert(T(2,2) == 1); assert(T(2,3) == 4);
assert(T(3,1) == 3); assert(T(3,2) == 2); assert(T(3,3) == 1);

% Test 3: non-square (more columns)
% toeplitz([1;2], [1 3 5 7]) -> [1 3 5 7; 2 1 3 5]
T = toeplitz([1;2], [1 3 5 7]);
assert(isequal(size(T), [2 4]));
assert(T(1,1) == 1); assert(T(1,2) == 3); assert(T(1,3) == 5); assert(T(1,4) == 7);
assert(T(2,1) == 2); assert(T(2,2) == 1); assert(T(2,3) == 3); assert(T(2,4) == 5);

% Test 4: non-square (more rows)
% toeplitz([1;2;3;4], [1 5]) -> [1 5; 2 1; 3 2; 4 3]
T = toeplitz([1;2;3;4], [1 5]);
assert(isequal(size(T), [4 2]));
assert(T(1,1) == 1); assert(T(1,2) == 5);
assert(T(2,1) == 2); assert(T(2,2) == 1);
assert(T(3,1) == 3); assert(T(3,2) == 2);
assert(T(4,1) == 4); assert(T(4,2) == 3);

% Test 5: scalar
T = toeplitz(7);
assert(T == 7);
assert(isequal(size(T), [1 1]));

% Test 6: complex symmetric (real first element)
% toeplitz([1 2+3i]) -> first row = [1 2+3i], first col = [1; 2-3i]
r = [1, 2+3i];
T = toeplitz(r);
assert(isequal(size(T), [2 2]));
assert(real(T(1,1)) == 1 && imag(T(1,1)) == 0);
assert(real(T(1,2)) == 2 && imag(T(1,2)) == 3);
assert(real(T(2,1)) == 2 && imag(T(2,1)) == -3);
assert(real(T(2,2)) == 1 && imag(T(2,2)) == 0);

% Test 7: complex nonsymmetric
% toeplitz([1+2i; 3+4i], [1+2i 5+6i])
c = [1+2i; 3+4i];
r = [1+2i, 5+6i];
T = toeplitz(c, r);
assert(isequal(size(T), [2 2]));
assert(real(T(1,1)) == 1 && imag(T(1,1)) == 2);
assert(real(T(1,2)) == 5 && imag(T(1,2)) == 6);
assert(real(T(2,1)) == 3 && imag(T(2,1)) == 4);
assert(real(T(2,2)) == 1 && imag(T(2,2)) == 2);

disp('SUCCESS');
