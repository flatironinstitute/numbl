% Test structfun

% Test 1: UniformOutput true (default) - returns column vector
S.a = [1 2 3];
S.b = [4 5];
S.c = [6 7 8 9];
A = structfun(@length, S);
assert(isequal(A, [3; 2; 4]));

% Test 2: UniformOutput false - returns struct
B = structfun(@(x) x * 2, S, 'UniformOutput', false);
assert(isequal(B.a, [2 4 6]));
assert(isequal(B.b, [8 10]));
assert(isequal(B.c, [12 14 16 18]));

% Test 3: Simple scalar function
S2.x = 10;
S2.y = 20;
S2.z = 30;
C = structfun(@(v) v + 1, S2);
assert(isequal(C, [11; 21; 31]));

fprintf('SUCCESS\n');
