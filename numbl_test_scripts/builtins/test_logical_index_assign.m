% Test logical indexing assignment with matrix
N = 4;
A = reshape(1:N^2, N, N);
B = reshape(101:100+N^2, N, N);
idx = logical(triu(ones(N)));
A(idx) = B(idx);

% Verify that upper-triangular elements were replaced
assert(A(1,1) == B(1,1));
assert(A(1,2) == B(1,2));
assert(A(2,1) == 2); % Lower-triangular should remain

fprintf('SUCCESS\n');
