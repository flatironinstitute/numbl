% Test nested end in indexing: D(ii(end:-1:3)) = ...
% The 'end' inside ii(...) should refer to ii's size, not D's size
N = 4;
D = reshape(1:N^2, N, N);
ii = (1:N+1:N^2).';
D(ii(end:-1:N-floor(N/2)+1)) = -D(ii(1:floor(N/2)));
assert(D(ii(end)) == -D(ii(1)));

% Also verify regular end still works
x = [10 20 30 40];
assert(x(end) == 40);
assert(x(end-1) == 30);

% Nested end on RHS as well
y = [1 2 3 4 5];
z = y(y(end));
assert(z == 5);

fprintf('SUCCESS\n');
