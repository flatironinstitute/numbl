% Deletion A(I) = [] and A(I,J,...) = [] (MATLAB rules).

% Linear deletion keeps a vector's orientation and makes anything else a row
A = magic(3); A(2) = []; assert(isequal(A, [8 4 1 5 9 6 7 2]));
A = magic(3); A(logical([1 0 1])) = []; assert(isequal(A, [3 1 5 9 6 7 2]));
A = magic(3); A([1 1 2]) = []; assert(isequal(A, [4 1 5 9 6 7 2]));
A = (1:3)'; A(2) = []; assert(isequal(A, [1; 3]));
A = (1:3)'; A([1 2 3]) = []; assert(isequal(size(A), [0 1]));
A = 1:3; A([1 2 3]) = []; assert(isequal(size(A), [1 0]));
A = 5; A(1) = []; assert(isequal(size(A), [1 0]));
R = reshape(1:24, 2, 3, 4); R(3) = []; assert(isequal(size(R), [1 23]));
A = magic(3); A(:) = []; assert(isequal(size(A), [0 0]));

% Deleting nothing: a numeric empty index still makes a matrix a row,
% an all-false mask leaves it alone, vectors are unchanged
A = magic(3); A([]) = []; assert(isequal(size(A), [1 9]));
A = magic(3); A(false) = []; assert(isequal(A, magic(3)));
A = magic(3); A(logical(zeros(3))) = []; assert(isequal(A, magic(3)));
A = (1:3)'; A([]) = []; assert(isequal(A, (1:3)'));
A = 1:3; A(zeros(1, 0)) = []; assert(isequal(A, 1:3));

% Subscript deletion removes slices along the one non-colon subscript
A = magic(3); A(:, 2) = []; assert(isequal(A, [8 6; 3 7; 4 2]));
A = magic(3); A(2, :) = []; assert(isequal(A, [8 1 6; 4 9 2]));
A = magic(3); A(:, [1 1]) = []; assert(isequal(A, [1 6; 5 7; 9 2]));
A = magic(3); A(:, logical([1 0 1])) = []; assert(isequal(A, [1; 5; 9]));
A = magic(3); A(:, logical([1 0 1 0])) = []; assert(isequal(A, [1; 5; 9]));
A = magic(3); A(1:3, :) = []; assert(isequal(size(A), [0 3]));
A = magic(3); A(:, :) = []; assert(isequal(size(A), [0 3]));
A = magic(3); A(:, []) = []; assert(isequal(A, magic(3)));
A = magic(3); A(:, :, 1) = []; assert(isequal(size(A), [3 3 0]));
A = magic(3); A(1, :, :) = []; assert(isequal(A, [3 5 7; 4 9 2]));
A = magic(3); for k = 1:3, A(:, 1) = []; end; assert(isequal(size(A), [3 0]));
A = magic(3); for k = 1:3, A(1, :) = []; end; assert(isequal(size(A), [0 3]));
R = reshape(1:24, 2, 3, 4); R(:, :, 2) = []; assert(isequal(size(R), [2 3 3]) && R(1, 1, 2) == 13);
R = reshape(1:24, 2, 3, 4); R(:, 2, :) = []; assert(isequal(size(R), [2 2 4]) && R(1, 2, 1) == 5);
% fewer subscripts than dims: deletion works on the collapsed view
R = reshape(1:24, 2, 3, 4); R(:, 2) = []; assert(isequal(size(R), [2 11]) && R(1, 2) == 5);
R = reshape(1:8, 2, 2, 2); R(:, :, 1) = []; assert(isequal(R, [5 7; 6 8]));

% '' deletes like []
A = 1:3; A(2) = ''; assert(isequal(A, [1 3]));

% Class is preserved
L = true(2, 2); L(:, 1) = []; assert(islogical(L) && isequal(size(L), [2 1]));
Z = [1i 2 3]; Z(1) = []; assert(isequal(Z, [2 3]));

% Errors
try
    A = magic(3); A(1, 2) = []; error('no error');
catch e
    assert(strcmp(e.message, 'A null assignment can have only one non-colon index.'));
end
try
    A = magic(3); A(1, [1 2 3]) = []; error('no error');
catch e
    assert(strcmp(e.message, 'A null assignment can have only one non-colon index.'));
end
try
    A = magic(3); A(:, 4) = []; error('no error');
catch e
    assert(strcmp(e.message, 'Matrix index is out of range for deletion.'));
end
try
    A = magic(3); A(10) = []; error('no error');
catch e
    assert(strcmp(e.message, 'Matrix index is out of range for deletion.'));
end
try
    A = magic(3); A(:, logical([1 0 1 1])) = []; error('no error');
catch e
    assert(strcmp(e.message, 'Matrix index is out of range for deletion.'));
end
% only a 0x0 RHS deletes
try
    A = 1:3; A(2) = zeros(1, 0); error('no error');
catch e
    assert(strcmp(e.message, 'Unable to perform assignment because the left and right sides have a different number of elements.'));
end

disp('SUCCESS');
