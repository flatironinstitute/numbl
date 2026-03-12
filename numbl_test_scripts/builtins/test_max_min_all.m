% Test max and min with 'all' flag
A = [1 5; 3 2];
assert(max(A, [], 'all') == 5);
assert(min(A, [], 'all') == 1);

% 3D-like matrix
B = [10 20 30; 40 50 60];
assert(max(B, [], 'all') == 60);
assert(min(B, [], 'all') == 10);

% Scalar
assert(max(7, [], 'all') == 7);
assert(min(7, [], 'all') == 7);

fprintf('SUCCESS\n');
