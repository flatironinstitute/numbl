% Test var() and std() with weight/normalization flag
% var(v, 0) = sample variance (N-1), var(v, 1) = population variance (N)
% std(v, 0) = sample std dev (N-1), std(v, 1) = population std dev (N)

v = [2 4 4 4 5 5 7 9];

% var with default (N-1)
assert(abs(var(v) - 32/7) < 1e-10);

% var(v, 0) should be same as var(v)
assert(abs(var(v, 0) - 32/7) < 1e-10);

% var(v, 1) should use N normalization
assert(abs(var(v, 1) - 4) < 1e-10);

% std with default (N-1)
assert(abs(std(v) - sqrt(32/7)) < 1e-10);

% std(v, 0) should be same as std(v)
assert(abs(std(v, 0) - sqrt(32/7)) < 1e-10);

% std(v, 1) should use N normalization
assert(abs(std(v, 1) - 2) < 1e-10);

disp('SUCCESS');
