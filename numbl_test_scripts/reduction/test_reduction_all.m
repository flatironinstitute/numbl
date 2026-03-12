% Test 'all' dimension option for reduction functions

M = [1 2; 3 4];

% sum with 'all'
assert(sum(M, 'all') == 10);

% prod with 'all'
assert(prod(M, 'all') == 24);

% mean with 'all'
assert(mean(M, 'all') == 2.5);

% 3D array
A = reshape(1:6, 2, 3);
assert(sum(A, 'all') == 21);
assert(prod(A, 'all') == 720);

% min/max with 'all' (already work, verify they still do)
assert(max(M, [], 'all') == 4);
assert(min(M, [], 'all') == 1);

disp('SUCCESS');
