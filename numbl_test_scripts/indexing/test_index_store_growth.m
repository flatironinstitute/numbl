% Indexed assignment: growth, scalar expansion, RHS shape rules, class
% preservation and errors (MATLAB rules).

% Linear growth follows the vector's orientation; empties and scalars
% grow as rows; matrices cannot grow linearly.
A = zeros(0, 1); A(3) = 1; assert(isequal(A, [0; 0; 1]));
A = zeros(1, 0); A(3) = 1; assert(isequal(A, [0 0 1]));
A = zeros(0, 3); A(2) = 1; assert(isequal(A, [0 1]));
A = zeros(3, 0); A(2) = 1; assert(isequal(A, [0 1]));
A = 5; A(3) = 1; assert(isequal(A, [5 0 1]));
A = (1:2)'; A(4) = 1; assert(isequal(A, [1; 2; 0; 1]));
A = []; for k = 1:4, A(end+1) = k; end; assert(isequal(A, 1:4));
try
    A = magic(3); A(12) = 1; error('no error');
catch e
    assert(strcmp(e.message, 'Attempt to grow array along ambiguous dimension.'));
end

% A logical mask grows the array only through a true element
A = 1:3; A(logical([0 0 0 0 1])) = 9; assert(isequal(A, [1 2 3 0 9]));
A = 1:3; A(logical([1 0 0 0 0])) = 9; assert(isequal(A, [9 2 3]));
A = (1:3)'; A(logical([0 0 0 0 1])) = 9; assert(isequal(A, [1; 2; 3; 0; 9]));

% A(:) = ... keeps the shape; the element count must match
A = zeros(2, 2); A(:) = 1:4; assert(isequal(A, [1 3; 2 4]));
A = magic(3); A(:) = magic(3)'; assert(isequal(A, magic(3)'));
A = []; A(:) = 5; assert(isequal(size(A), [0 0]));
try
    A = zeros(2, 2); A(:) = 1:3; error('no error');
catch e
    assert(strcmp(e.message, 'Unable to perform assignment because the left and right sides have a different number of elements.'));
end

% Linear assignment only needs matching element counts
A = 1:4; A([1 2]) = [7; 8]; assert(isequal(A, [7 8 3 4]));
A = zeros(2, 3); A([1 2; 3 4]) = [1 2 3 4]; assert(isequal(A, [1 2 0; 3 4 0]));
A = 1:3; A([]) = 5; assert(isequal(A, 1:3));
A = [2 1 3]; A(A) = [10 20 30]; assert(isequal(A, [20 10 30]));

% Subscript assignment compares non-singleton dimensions
A = magic(3); A(1, :) = [1; 2; 3]; assert(isequal(A(1, :), [1 2 3]));
A = magic(3); A(:, 1) = [1 2 3]; assert(isequal(A(:, 1), [1; 2; 3]));
A = zeros(2, 3, 2); A(1, :, :) = ones(3, 2); assert(isequal(squeeze(A(1, :, :)), ones(3, 2)));
A = zeros(2, 3, 2); A(:, 1, :) = ones(2, 2); assert(isequal(A(:, 1, 2), [1; 1]));
try
    A = magic(3); A(1:2, 1:3) = ones(3, 2); error('no error');
catch e
    assert(strcmp(e.message, 'Unable to perform assignment because the size of the left side is 2-by-3 and the size of the right side is 3-by-2.'));
end
try
    A = magic(3); A(:, :) = 1:9; error('no error');
catch e
    assert(strcmp(e.message, 'Unable to perform assignment because the size of the left side is 3-by-3 and the size of the right side is 1-by-9.'));
end
try
    A = zeros(2, 3, 2); A(:, 1, :) = ones(1, 4); error('no error');
catch e
    assert(strcmp(e.message, 'Unable to perform assignment because the size of the left side is 2-by-1-by-2 and the size of the right side is 1-by-4.'));
end

% Subscript growth pads with zeros
A = magic(3); A(5, 5) = 1; assert(isequal(size(A), [5 5]) && A(5, 5) == 1 && A(4, 1) == 0);
A = 1:3; A(2, :) = [4 5 6]; assert(isequal(A, [1 2 3; 4 5 6]));
A = 1:3; A(:, 5) = 7; assert(isequal(A, [1 2 3 0 7]));
A = 1:3; A(1, 1, 2) = 9; assert(isequal(size(A), [1 3 2]) && A(1, 1, 2) == 9);
A = zeros(2, 2); A(:, :, 2) = ones(2, 2); assert(isequal(size(A), [2 2 2]));
R = reshape(1:24, 2, 3, 4); R(:, :, 5) = 1; assert(isequal(size(R), [2 3 5]) && R(2, 3, 5) == 1);
% growing a collapsed trailing dimension is ambiguous; other dims may grow
A = zeros(2, 3, 4); A(3, 12) = 1; assert(isequal(size(A), [3 3 4]) && A(3, 3, 4) == 1);
try
    A = zeros(2, 3, 4); A(1, 13) = 1; error('no error');
catch e
    assert(strcmp(e.message, 'Attempt to grow array along ambiguous dimension.'));
end

% Assigning into [] sizes colons from the RHS
A = []; A(:, 1) = [1; 2; 3]; assert(isequal(A, [1; 2; 3]));
A = []; A(:, 2) = [1 2 3]; assert(isequal(A, [0 1; 0 2; 0 3]));
A = []; A(2, :) = [1 2 3]; assert(isequal(A, [0 0 0; 1 2 3]));
A = []; A(:, :) = ones(2, 3); assert(isequal(A, ones(2, 3)));
A = []; A(:, :) = ones(1, 3); assert(isequal(A, ones(1, 3)));
A = []; A(:, :, 2) = ones(2, 3); assert(isequal(size(A), [2 3 2]));
A = []; A(1, :, :) = ones(2, 3); assert(isequal(size(A), [1 2 3]));
A = []; A(:, 1, :) = ones(2, 3); assert(isequal(size(A), [2 1 3]));
A = []; A(:, 3) = 5; assert(isequal(A, [0 0 5]));
% a non-0x0 empty keeps its known extents
A = zeros(0, 3); A(1, :) = [1 2 3]; assert(isequal(A, [1 2 3]));
try
    A = zeros(0, 3); A(:, 1) = [1; 2]; error('no error');
catch e
    assert(strcmp(e.message, 'Unable to perform assignment because the size of the left side is 0-by-1 and the size of the right side is 2-by-1.'));
end

% Logical arrays stay logical; values are converted
L = true(1, 3); L(2) = 5; assert(islogical(L) && isequal(L, [true true true]));
L = true(1, 3); L(2) = 0; assert(islogical(L) && isequal(L, [true false true]));
L = true(1, 3); L(5) = true; assert(islogical(L) && isequal(L, [true true true false true]));
L = true; L(3) = 1; assert(islogical(L) && isequal(L, [true false true]));
try
    L = true(1, 3); L(2) = NaN; error('no error');
catch e
    assert(strcmp(e.message, 'NaN values cannot be converted to logicals.'));
end
% a double target stays double
x = []; x(3) = true; assert(isa(x, 'double') && isequal(x, [0 0 1]));
x = 1:3; x(2) = 'A'; assert(isa(x, 'double') && isequal(x, [1 65 3]));

% Complex values
x = [1 2 3]; x(5) = 1i; assert(isequal(x, [1 2 3 0 1i]));
x = [1 2 3]; x(2) = 2 + 3i; assert(isequal(x, [1 2+3i 3]));

% Invalid subscripts
try
    A = magic(3); A(0) = 1; error('no error');
catch e
    assert(strcmp(e.message, 'Array indices must be positive integers or logical values.'));
end
try
    A = magic(3); A(1, 0) = 1; error('no error');
catch e
    assert(strcmp(e.message, 'Index in position 2 is invalid. Array indices must be positive integers or logical values.'));
end

% Copy-on-write: the source of a copy is unchanged
P = [1 2 3]; Q = P; Q(2) = 9; assert(isequal(P, [1 2 3]) && isequal(Q, [1 9 3]));
P = [1 2 3]; Q = P; Q(5) = 9; assert(isequal(P, [1 2 3]));

disp('SUCCESS');
