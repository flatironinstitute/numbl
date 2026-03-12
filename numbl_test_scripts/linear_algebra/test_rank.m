% Test rank builtin

% Full rank matrix
A = eye(3);
assert(rank(A) == 3);

% Rank-deficient matrix (third column = 2 * second column)
B = [3 2 4; -1 1 2; 9 5 10];
assert(rank(B) == 2);

% Scalar
assert(rank(5) == 1);
assert(rank(0) == 0);

% Complex scalar
assert(rank(1 + 2i) == 1);
assert(rank(0 + 0i) == 0);

% Zero matrix
Z = zeros(3, 3);
assert(rank(Z) == 0);

% Row vector
assert(rank([1 2 3]) == 1);

% Column vector
assert(rank([1; 2; 3]) == 1);

disp('SUCCESS');
