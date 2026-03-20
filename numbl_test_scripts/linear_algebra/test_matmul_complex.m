% Test complex matrix multiplication via native zgemm addon.

%% Basic 2x2 complex
A = [1+2i 3+4i; 5+6i 7+8i];
B = [1+1i 2+2i; 3+3i 4+4i];
C = A*B;
expected = [-4+24i -6+34i; -4+56i -6+82i];
assert(max(abs(C(:) - expected(:))) < 1e-10, '2x2 complex matmul');

%% Real * complex
Ar = [1 2; 3 4];
Bc = [1+1i 2+2i; 3+3i 4+4i];
C2 = Ar * Bc;
expected2 = [7+7i 10+10i; 15+15i 22+22i];
assert(max(abs(C2(:) - expected2(:))) < 1e-10, 'real * complex');

%% Complex * real
C3 = A * [1 0; 0 1];
assert(max(abs(C3(:) - A(:))) < 1e-10, 'complex * identity');

%% Non-square
M = complex(randn(5,3), randn(5,3));
N = complex(randn(3,4), randn(3,4));
P = M * N;
assert(all(size(P) == [5 4]), 'non-square size');

% Verify against element-wise: P(1,1) = sum(M(1,:).*N(:,1).')
val = sum(M(1,:) .* N(:,1).');
assert(abs(P(1,1) - val) < 1e-10, 'non-square value check');

%% Larger matrix — check A*inv(A) ≈ I
n = 50;
X = complex(randn(n), randn(n));
Y = inv(X);
I_approx = X * Y;
I_exact = eye(n);
err = max(abs(I_approx(:) - I_exact(:)));
assert(err < 1e-8, sprintf('A*inv(A) should be ~I, error=%.2e', err));

%% Vector inner product: row * col
r = [1+1i 2+2i 3+3i];
c = [1-1i; 2-2i; 3-3i];
val2 = r * c;
expected_val = (1+1i)*(1-1i) + (2+2i)*(2-2i) + (3+3i)*(3-3i);
assert(abs(val2 - expected_val) < 1e-10, 'complex dot product');

disp('SUCCESS');
