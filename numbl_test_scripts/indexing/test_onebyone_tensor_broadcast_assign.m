% A 1-element RHS must broadcast across an indexed-assignment slice
% (MATLAB scalar expansion) regardless of how the 1-element value is
% stored. A 1x1 tensor produced by a 2-D slice (e.g. x(:,k), the `dealm`
% idiom) previously failed with "Subscripted assignment dimension
% mismatch" while a literal scalar worked.

r = [2 1 0 9];

% 1x1 tensor from a 2-D column slice -> broadcast into a row slice
A = zeros(2, 4);
A(1, 1:4) = r(:, 1);
assert(isequal(A(1, :), [2 2 2 2]), 'row-slice broadcast of x(:,k) failed');
assert(isequal(A(2, :), [0 0 0 0]), 'unrelated row disturbed');

% computed 1x1 tensor -> broadcast into a column slice
B = zeros(3, 2);
B(1:3, 1) = zeros(1, 1) + 5;
assert(isequal(B(:, 1), [5; 5; 5]), 'col-slice broadcast of 1x1 tensor failed');

% 1x1 tensor into a 2-D block
C = zeros(3, 3);
C(1:2, 2:3) = r(:, 1);
assert(isequal(C(1:2, 2:3), [2 2; 2 2]), '2-D block broadcast failed');

% complex 1x1 tensor broadcasts real+imag parts
z = complex(r(:, 1), r(:, 2));   % 2 + 1i, stored as 1x1 tensor
D = zeros(1, 3);
D(1, 1:3) = z;
assert(isequal(D, [2+1i, 2+1i, 2+1i]), 'complex 1x1 broadcast failed');

disp('SUCCESS');
