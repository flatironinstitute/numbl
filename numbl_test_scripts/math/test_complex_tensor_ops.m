% Test complex tensor operations that lose imaginary data

% Bug 1: for-loop over complex matrix loses imaginary parts
A = [1+2i 3+4i; 5+6i 7+8i];
cols = {};
k = 1;
for col = A
    cols{k} = col;
    k = k + 1;
end
assert(cols{1}(1) == 1+2i, 'for complex col1(1)');
assert(cols{1}(2) == 5+6i, 'for complex col1(2)');
assert(cols{2}(1) == 3+4i, 'for complex col2(1)');
assert(cols{2}(2) == 7+8i, 'for complex col2(2)');

% Bug 2: cat along dim 3 with complex arrays loses imaginary parts
B = [1+1i 2+2i];
C = [3+3i 4+4i];
D = cat(3, B, C);
assert(isequal(size(D), [1 2 2]), 'cat3 complex size');
assert(D(1,1,1) == 1+1i, 'cat3 complex (1,1,1)');
assert(D(1,2,1) == 2+2i, 'cat3 complex (1,2,1)');
assert(D(1,1,2) == 3+3i, 'cat3 complex (1,1,2)');
assert(D(1,2,2) == 4+4i, 'cat3 complex (1,2,2)');

% Bug 3: sign on complex tensors ignores imaginary parts
E = [3+4i, -1+0i, 0+2i];
r = sign(E);
assert(abs(r(1) - (0.6+0.8i)) < 1e-10, 'sign complex tensor (1)');
assert(abs(r(2) - (-1)) < 1e-10, 'sign complex tensor (2)');
assert(abs(r(3) - 1i) < 1e-10, 'sign complex tensor (3)');

disp('SUCCESS');
