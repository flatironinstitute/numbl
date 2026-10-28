% Linear-algebra builtins keep the imaginary part of complex inputs.
tol = 1e-12;

% cross
c = cross([1i 0 0], [0 1 0]);
assert(isequal(c, [0 0 1i]));
c = cross([1+2i, 3, -1i], [2, 1i, 4-1i]);
assert(isequal(c, [11-3i, -6-9i, -8+1i]));
c = cross([1 2 3; 4 5 6i; 7 8 9].', [1 0 0; 0 1 0; 0 0 1i].', 1);
assert(isequal(c, [0 -6i 8i; 3 0 -7i; -2 4 0]));
c = cross([1 2 3; 4 5 6], [1i 0 0; 0 1 0], 2);
assert(isequal(c, [0 3i -2i; -6 0 4]));
c = cross([1i 0 0], [1i 0 0]);
assert(isreal(c) && isequal(c, [0 0 0]));

% nnz
assert(nnz([1i 0 0]) == 1);
assert(nnz([1i 2 0; 0 0 -1i]) == 3);
assert(nnz(2i) == 1);

% kron
k = kron([1 1i], [1; 2i]);
assert(isequal(k, [1 1i; 2i -2]));
assert(isequal(kron(1i, [1 2]), [1i 2i]));
assert(isreal(kron([1 2], [3; 4])));

% blkdiag
b = blkdiag(1i, [2 3i]);
assert(isequal(b, [1i 0 0; 0 2 3i]));
assert(isreal(blkdiag(1, [2 3])));

% pagetranspose
X = cat(3, [1+1i 2; 3 4-2i], [2 1i; 0 1]);
Y = pagetranspose(X);
assert(isequal(Y(:,:,1), [1+1i 3; 2 4-2i]));
assert(isequal(Y(:,:,2), [2 0; 1i 1]));

% pagemtimes
A = [1+1i 2; 3 4-2i];
B = [2 1i; 0 1];
assert(isequal(pagemtimes(A, B), A*B));
Z = pagemtimes(X, 'ctranspose', B, 'none');
assert(isequal(Z(:,:,1), A'*B));
assert(isequal(Z(:,:,2), B'*B));
Z = pagemtimes(X, 'transpose', X, 'none');
assert(isequal(Z(:,:,1), A.'*A));
assert(isequal(Z(:,:,2), B.'*B));

% pinv (complex matrices need the LAPACK addon for the complex SVD)
assert(abs(pinv(2i) - (-0.5i)) < tol);
has_lapack = true;
try
  svd([1+1i, 0; 0, 1]);
catch e
  if isempty(strfind(e.message, 'requires LAPACK'))
    rethrow(e);
  end
  has_lapack = false;
end
if has_lapack
  P = pinv(A);
  assert(norm(P - inv(A)) < tol);
  M = [1 1i; 2 2i; 0 1];
  P = pinv(M);
  assert(norm(P - (M'*M) \ M') < tol);
end

disp('SUCCESS')
