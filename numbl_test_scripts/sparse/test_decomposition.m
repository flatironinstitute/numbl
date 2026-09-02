% decomposition objects: factor once, solve many times with \ and /.
rng(6);
n = 8; e = ones(n, 1); D = spdiags([e -2*e e], -1:1, n, n);
A = 4*speye(n^2) - (kron(speye(n), D) + kron(D, speye(n)));
m = n^2;
b = randn(m, 1);
Bm = randn(m, 3);

% auto type picks Cholesky for a symmetric positive definite matrix
dA = decomposition(A);
assert(isequal(class(dA), 'decomposition'));
assert(strcmp(dA.Type, 'chol'));
assert(isequal(dA.MatrixSize, [m m]));
x = dA \ b;
assert(isequal(size(x), [m 1]));
assert(norm(A*x - b) / norm(b) < 1e-10);
X = dA \ Bm;
assert(norm(A*X - Bm, 1) / norm(Bm, 1) < 1e-10);
y = b' / dA;
assert(isequal(size(y), [1 m]));
assert(norm(y*A - b') / norm(b) < 1e-10);

% explicit types
dL = decomposition(A, 'lu');
assert(strcmp(dL.Type, 'lu'));
assert(norm(A*(dL \ b) - b) / norm(b) < 1e-10);
dC = decomposition(A, 'chol');
assert(norm(A*(dC \ b) - b) / norm(b) < 1e-10);

% nonsymmetric matrix goes through LU; / solves with the transpose
N = A + 0.4*spdiags(ones(m, 1), 1, m, m);
dN = decomposition(N);
assert(strcmp(dN.Type, 'lu'));
xn = dN \ b;
assert(norm(N*xn - b) / norm(b) < 1e-10);
yn = Bm' / dN;
assert(norm(yn*N - Bm', 1) / norm(Bm, 1) < 1e-10);

% triangular
R = chol(A);
dR = decomposition(R, 'triangular');
assert(strcmp(dR.Type, 'triangular'));
assert(norm(R*(dR \ b) - b) / norm(b) < 1e-10);
dRa = decomposition(R);
assert(strcmp(dRa.Type, 'triangular'));
assert(norm((b' / dRa)*R - b') / norm(b) < 1e-10);
dLt = decomposition(R', 'triangular', 'lower');
assert(norm(R'*(dLt \ b) - b) / norm(b) < 1e-10);

% dense input (MATLAB's auto type for a narrow-band dense matrix is
% 'banded'; numbl reports the factorization it actually used)
Ad = [4 1 0; 1 4 1; 0 1 4];
dAd = decomposition(Ad);
assert(norm(dAd \ [1; 2; 3] - Ad \ [1; 2; 3]) < 1e-12);
Nd = [4 1 0; 2 4 1; 0 1 4];
dNd = decomposition(Nd, 'lu');
assert(norm(dNd \ [1; 2; 3] - Nd \ [1; 2; 3]) < 1e-12);
assert(norm([1 2 3] / dNd - [1 2 3] / Nd) < 1e-12);

% option handling and conditioning
dCC = decomposition(A, 'CheckCondition', false);
assert(~dCC.CheckCondition);
assert(~isIllConditioned(dA));
dIll = decomposition([1e-20 0; 0 1], 'lu', 'CheckCondition', false);
assert(isIllConditioned(dIll));
threw = false;
try
    decomposition(A - 6*speye(m), 'chol');
catch
    threw = true;
end
assert(threw);
disp('SUCCESS')
