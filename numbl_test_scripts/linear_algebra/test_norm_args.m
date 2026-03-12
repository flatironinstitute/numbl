% Test norm with string arguments and matrix norm modes

%% Vector norm with string arguments
v = [1, -3, 2];

% norm(v, 'inf') should equal norm(v, Inf)
assert(norm(v, 'inf') == 3);
assert(norm(v, Inf) == 3);

% norm(v, -Inf)
assert(norm(v, -Inf) == 1);

% norm(v, 'fro') should equal norm(v, 2) for vectors
assert(abs(norm(v, 'fro') - norm(v, 2)) < 1e-10);

%% Matrix norms
A = [1 2; 3 4];

% norm(A) default is 2-norm (largest singular value), not Frobenius
% For [1 2; 3 4], 2-norm is ~5.465, Frobenius is ~5.477
assert(abs(norm(A) - 5.46498570421904) < 1e-6);

% norm(A, 1) = max column sum
assert(norm(A, 1) == 6);

% norm(A, Inf) = max row sum
assert(norm(A, Inf) == 7);

% norm(A, 'inf') should equal norm(A, Inf)
assert(norm(A, 'inf') == 7);

% norm(A, 'fro') = Frobenius norm
assert(abs(norm(A, 'fro') - sqrt(30)) < 1e-10);

disp('SUCCESS');
