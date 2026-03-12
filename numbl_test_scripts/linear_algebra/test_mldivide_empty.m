% Test mldivide (\) with empty matrices

% 0x0 \ 0x0 should give 0x0
A = zeros(0,0);
B = zeros(0,0);
X = A \ B;
assert(isempty(X));
assert(isequal(size(X), [0 0]));

% 0x0 \ 0x5 should give 0x5
A = zeros(0,0);
B = zeros(0,5);
X = A \ B;
assert(isempty(X));
assert(isequal(size(X), [0 5]));

% 3x0 \ 3x2 should give 0x2
A = zeros(3,0);
B = zeros(3,2);
X = A \ B;
assert(isempty(X));
assert(isequal(size(X), [0 2]));

% 0x3 \ 0x2 should give 3x2
A = zeros(0,3);
B = zeros(0,2);
X = A \ B;
assert(isequal(size(X), [3 2]));
assert(all(X(:) == 0));

fprintf('SUCCESS\n');
