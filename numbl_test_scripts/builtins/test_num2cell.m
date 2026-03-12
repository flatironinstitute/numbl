% Test num2cell builtin

% Basic: convert matrix to cell array
A = [1 2 3; 4 5 6];
C = num2cell(A);
assert(iscell(C));
assert(isequal(size(C), [2, 3]));
assert(C{1,1} == 1);
assert(C{2,1} == 4);
assert(C{1,2} == 2);
assert(C{2,3} == 6);

% Scalar
C2 = num2cell(5);
assert(iscell(C2));
assert(C2{1} == 5);

% Along dimension 1 (columns become cells)
A = [1 2; 3 4; 5 6];
C3 = num2cell(A, 1);
assert(isequal(size(C3), [1, 2]));
assert(isequal(C3{1}, [1; 3; 5]));
assert(isequal(C3{2}, [2; 4; 6]));

% Along dimension 2 (rows become cells)
C4 = num2cell(A, 2);
assert(isequal(size(C4), [3, 1]));
assert(isequal(C4{1}, [1 2]));
assert(isequal(C4{2}, [3 4]));
assert(isequal(C4{3}, [5 6]));

% Vector
v = [10 20 30];
C5 = num2cell(v);
assert(isequal(size(C5), [1, 3]));
assert(C5{1} == 10);
assert(C5{3} == 30);

disp('SUCCESS');
