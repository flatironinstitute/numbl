% Test std/var with 'all' dimension flag
% MATLAB supports std(A, w, 'all') and var(A, w, 'all')

A = [1 2; 3 4];

% std with 'all' (w=0, normalize by N-1)
s = std(A, 0, 'all');
assert(abs(s - 1.2910) < 1e-3, 'std(A, 0, all) failed');

% var with 'all' (w=0, normalize by N-1)
v = var(A, 0, 'all');
assert(abs(v - 1.6667) < 1e-3, 'var(A, 0, all) failed');

% std with 'all' (w=1, normalize by N)
s1 = std(A, 1, 'all');
assert(abs(s1 - 1.1180) < 1e-3, 'std(A, 1, all) failed');

% var with 'all' (w=1, normalize by N)
v1 = var(A, 1, 'all');
assert(abs(v1 - 1.25) < 1e-3, 'var(A, 1, all) failed');

% 3D array
B = reshape(1:8, [2 2 2]);
sb = std(B(:), 0);
sb_all = std(B, 0, 'all');
assert(abs(sb - sb_all) < 1e-10, 'std 3D all should match std of flattened');

disp('SUCCESS');
