% Test that diag() returns correct shape for empty diagonal extraction
% MATLAB returns [0, 1] (empty column vector), not [0, 0]

A = zeros(2, 3);
d = diag(A, 5);
s = size(d);
assert(s(1) == 0, 'diag empty: row count should be 0');
assert(s(2) == 1, 'diag empty: col count should be 1');

B = zeros(3, 2);
d2 = diag(B, -5);
s2 = size(d2);
assert(s2(1) == 0, 'diag empty neg offset: row count should be 0');
assert(s2(2) == 1, 'diag empty neg offset: col count should be 1');

disp('SUCCESS');
