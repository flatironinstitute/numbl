% Test rcond builtin

% Identity matrix: perfectly conditioned
assert(abs(rcond(eye(3)) - 1) < 1e-10);

% Scaled identity is still perfectly conditioned (unlike det, which is tiny)
assert(abs(rcond(eye(5) * 0.01) - 1) < 1e-10);

% Diagonal matrix [1 0; 0 2]: rcond = 0.5
assert(abs(rcond([1 0; 0 2]) - 0.5) < 1e-10);

% Well-conditioned matrix [2 1; 1 3]: rcond = 0.3125
assert(abs(rcond([2 1; 1 3]) - 0.3125) < 1e-10);

% Scalars: nonzero is well conditioned, zero is singular
assert(rcond(5) == 1);
assert(rcond(0) == 0);

% Complex scalar
assert(rcond(3 + 4i) == 1);
assert(rcond(0 + 0i) == 0);

% Singular matrix: rcond = 0 (no error)
assert(rcond([1 2; 2 4]) == 0);

% Zero matrix: singular
assert(rcond(zeros(3)) == 0);

% Badly conditioned Hilbert matrix: small reciprocal condition number.
% MATLAB's LU-based estimate is 2.8286e-14; the inverse-based value agrees
% to a few significant figures.
n = 10;
H = zeros(n, n);
for i = 1:n
  for j = 1:n
    H(i, j) = 1 / (i + j - 1);
  end
end
h = rcond(H);
assert(h > 0);
assert(abs(h - 2.8286e-14) < 1e-15);

% Empty matrix
assert(rcond([]) == Inf);

% Complex matrix matches 1/cond(A,1)
Z = [1+1i, 2; 3, 4-1i];
assert(abs(rcond(Z) - 1 / cond(Z, 1)) < 1e-12);

disp('SUCCESS');
