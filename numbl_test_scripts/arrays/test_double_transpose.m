% Test double transpose (consecutive '' after expression)

% Double ctranspose on matrix
M = [1 2; 3 4];
R = M'';
assert(isequal(R, M));

% Double ctranspose on vector
v = [1 2 3];
v2 = v'';
assert(isequal(v2, v));

% Double ctranspose on scalar
x = 5'';
assert(x == 5);

% Double ctranspose on complex matrix
C = [1+1i, 2; 3, 4-1i];
R2 = C'';
assert(isequal(R2, C));

% Triple transpose (should also work)
M3 = [1 2; 3 4]''';
assert(isequal(M3, M'));

disp('SUCCESS');
