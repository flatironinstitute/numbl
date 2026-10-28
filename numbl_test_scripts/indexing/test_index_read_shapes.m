% Result shapes and errors of A(I) / A(I,J,...) reads (MATLAB rules).

A = magic(3);
v = 1:3;
w = (1:3)';

% Linear index: result takes the index's shape...
assert(isequal(size(A([1 2; 3 4])), [2 2]));
assert(isequal(size(A([1; 2])), [2 1]));
assert(isequal(size(A([1 2])), [1 2]));
% ...except a vector indexed by a vector keeps its own orientation
assert(isequal(size(v([1; 2])), [1 2]));
assert(isequal(size(w([1 2])), [2 1]));
% a matrix index keeps its shape even on a vector
assert(isequal(size(v([1 2; 3 1])), [2 2]));
assert(isequal(size(w([1 2; 3 1])), [2 2]));
% A(:) is a column
R = reshape(1:24, 2, 3, 4);
assert(isequal(size(R(:)), [24 1]));
assert(isequal(R(:)', 1:24));

% Logical masks act like find(mask)
assert(isequal(A(logical([1 0 1])), [8 4]));
assert(isequal(A(logical([1; 0; 1])), [8; 4]));
assert(isequal(size(A(logical([1 0; 0 1]))), [2 1]));
assert(isequal(w(logical([1 0 1])), [1; 3]));
assert(isequal(v(logical([1; 0; 1])), [1 3]));
% a mask longer than numel is fine if the tail is false
assert(isequal(A(logical([1 0 0 0 0 0 0 0 0 0 0])), 8));
assert(isequal(v(logical([1 0 0 0 0])), 1));

% Empty results
assert(isequal(size(A(false)), [0 0]));
assert(isequal(size(A(logical([0 0 0]))), [1 0]));
assert(isequal(size(A(logical([0; 0; 0]))), [0 1]));
assert(isequal(size(A(logical(zeros(2, 2)))), [0 1]));
assert(isequal(size(v(logical([0 0 0]))), [1 0]));
assert(isequal(size(w(logical([0 0 0]))), [0 1]));
assert(isequal(size(v(false)), [0 0]));
assert(isequal(size(w(zeros(1, 0))), [0 1]));
assert(isequal(size(v(zeros(0, 1))), [1 0]));
assert(isequal(size(w([])), [0 0]));
assert(isequal(size(A([])), [0 0]));
assert(isequal(size(A(zeros(1, 0))), [1 0]));
assert(isequal(size(A(zeros(0, 1))), [0 1]));
assert(isequal(size(A(:, [])), [3 0]));
assert(isequal(size(A([], :)), [0 3]));
assert(isequal(size(A([], [])), [0 0]));
assert(isequal(size(A(zeros(1, 0), 2)), [0 1]));
assert(isequal(size(R(:, :, [])), [2 3 0]));
assert(isequal(size(R(:, 1, [])), [2 1 0]));

% Scalars are 1x1 arrays
x = 5;
assert(isequal(x([1 1 1]), [5 5 5]));
assert(isequal(x([1; 1]), [5; 5]));
assert(isequal(x(:), 5));
assert(isequal(size(x([])), [0 0]));
assert(isequal(size(x(zeros(1, 0))), [1 0]));
assert(isequal(size(x(false)), [0 0]));
assert(isequal(x(1, 1, 1), 5));
b = true;
assert(islogical(b([1 1])) && isequal(b([1 1]), [true true]));
z = 1i;
assert(isequal(z([1 1]), [1i 1i]));

% Subscript indexing
M = magic(4);
assert(isequal(M(2:3, [1 4]), [5 8; 9 12]));
assert(isequal(M([2 2], :), [5 11 10 8; 5 11 10 8]));
% fewer subscripts than dims: the last one spans the trailing dims
assert(isequal(R(:, 5), [9; 10]));
assert(isequal(size(R(2, :)), [1 12]));
assert(isequal(R(1, :, 2), [7 9 11]));
assert(isequal(size(R(1, 2, :)), [1 1 4]));
assert(isequal(squeeze(R(1, 2, :))', [3 9 15 21]));
assert(isequal(size(R(:, :, :, 1)), [2 3 4]));
assert(R(1, 1, 1, 1, 1) == 1);
assert(isequal(R([1 2; 3 4]), [1 2; 3 4]));
assert(isequal(R([1 3; 2 4]), [1 3; 2 4]));
assert(isequal(R(1:5), 1:5));

% Complex: an all-zero imaginary part is dropped
Z = [1+2i 3; 4 5];
assert(isreal(Z(:, 2)));
assert(isreal(Z(2)));
assert(~isreal(Z(:, 1)));

% Logical class is preserved
L = true(2, 2);
assert(islogical(L(:, 1)) && isequal(size(L(:, 1)), [2 1]));
assert(islogical(L(1)));

% Errors
try
    A(10); error('no error');
catch e
    assert(strcmp(e.message, 'Index exceeds the number of array elements. Index must not exceed 9.'));
end
try
    A(0); error('no error');
catch e
    assert(strcmp(e.message, 'Array indices must be positive integers or logical values.'));
end
try
    A(1.5); error('no error');
catch e
    assert(strcmp(e.message, 'Array indices must be positive integers or logical values.'));
end
try
    A(4, 1); error('no error');
catch e
    assert(strcmp(e.message, 'Index in position 1 exceeds array bounds. Index must not exceed 3.'));
end
try
    A(1, 0); error('no error');
catch e
    assert(strcmp(e.message, 'Index in position 2 is invalid. Array indices must be positive integers or logical values.'));
end
try
    A(1, 1, 2); error('no error');
catch e
    assert(strcmp(e.message, 'Index in position 3 exceeds array bounds. Index must not exceed 1.'));
end
try
    R(1, 13); error('no error');
catch e
    assert(strcmp(e.message, 'Index in position 2 exceeds array bounds.'));
end
try
    A(logical([0 0 0 0 0 0 0 0 0 1])); error('no error');
catch e
    assert(strcmp(e.message, 'The logical indices contain a true value outside of the array bounds.'));
end

disp('SUCCESS');
