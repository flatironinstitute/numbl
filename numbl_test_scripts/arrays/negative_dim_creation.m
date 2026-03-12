% Test that array constructors with negative dimensions return empty arrays
% (matching MATLAB behavior where negative dims are treated as 0)

% ones with negative first dimension
A = ones(-1, 1);
assert(isempty(A));
assert(size(A, 1) == 0);
assert(size(A, 2) == 1);

% ones with negative computed dimension (the original bug trigger)
n = 1;
dg = .5*ones(n - 2, 1);
assert(isempty(dg));
assert(size(dg, 1) == 0);
assert(size(dg, 2) == 1);

% zeros with negative dimension
B = zeros(-3, 4);
assert(isempty(B));
assert(size(B, 1) == 0);
assert(size(B, 2) == 4);

% zeros with both negative
C = zeros(-2, -3);
assert(isempty(C));
assert(size(C, 1) == 0);
assert(size(C, 2) == 0);

% ones single negative arg: ones(-1) should give 0x0
D = ones(-1);
assert(isempty(D));

disp('SUCCESS');
