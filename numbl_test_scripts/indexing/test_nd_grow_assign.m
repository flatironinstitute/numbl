% Test N-D subscripted assignment that creates/grows arrays

% Create a 3D array via subscripted assignment to undefined variable
X(:,:,1) = zeros(3, 4);
assert(isequal(size(X), [3, 4]));
assert(isequal(X, zeros(3, 4)));

% Grow by assigning to a new 3rd-dimension slice
X(:,:,2) = ones(3, 4);
assert(isequal(size(X), [3, 4, 2]));
assert(isequal(X(:,:,1), zeros(3, 4)));
assert(isequal(X(:,:,2), ones(3, 4)));

% Create with non-trivial values
clear Y;
Y(:,:,1) = [1 2; 3 4];
assert(isequal(Y, [1 2; 3 4]));

% Grow in the 3rd dim with a gap (should zero-fill)
clear Z;
Z(:,:,2) = [5 6; 7 8];
assert(isequal(size(Z), [2, 2, 2]));
assert(isequal(Z(:,:,1), zeros(2, 2)));
assert(Z(1,1,2) == 5);

fprintf('SUCCESS\n');
