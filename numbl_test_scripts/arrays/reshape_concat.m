% reshape, concatenation, linspace, colon ranges

% reshape
A = [1, 2, 3, 4, 5, 6];
B = reshape(A, 2, 3);
assert(size(B, 1) == 2)
assert(size(B, 2) == 3)
assert(B(1,1) == 1)
assert(B(2,1) == 2)
assert(B(1,2) == 3)

% horzcat
c = [1, 2, 3];
d = [4, 5, 6];
e = [c, d];
assert(length(e) == 6)
assert(e(4) == 4)

% vertcat
f = [c; d];
assert(size(f, 1) == 2)
assert(size(f, 2) == 3)
assert(f(2, 1) == 4)

% linspace
v = linspace(0, 1, 5);
assert(length(v) == 5)
assert(abs(v(1)) < 1e-6)
assert(abs(v(5) - 1) < 1e-6)
assert(abs(v(3) - 0.5) < 1e-5)

% numel
M = ones(3, 4);
assert(numel(M) == 12)

% colon as vector
r = 1:5;
assert(length(r) == 5)
assert(r(1) == 1)
assert(r(5) == 5)

% reshape with size vector [m n]
C = reshape(A, [2, 3]);
assert(size(C, 1) == 2)
assert(size(C, 2) == 3)
assert(C(1,1) == 1)
assert(C(2,3) == 6)

% reshape with auto dimension []
D = reshape(A, 2, []);
assert(size(D, 1) == 2)
assert(size(D, 2) == 3)

E = reshape(ones(10,10), 2, 2, []);
assert(size(E, 1) == 2)
assert(size(E, 2) == 2)
assert(size(E, 3) == 25)

disp('SUCCESS')
