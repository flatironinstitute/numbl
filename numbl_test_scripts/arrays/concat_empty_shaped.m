% Test concatenation of empty matrices with non-trivial shapes

% Vertical concatenation of Mx0 matrices
A = zeros(30, 0);
B = zeros(30, 0);
C = [A; B];
assert(size(C, 1) == 60);
assert(size(C, 2) == 0);

% Horizontal concatenation of 0xN matrices
A = zeros(0, 30);
B = zeros(0, 30);
C = [A, B];
assert(size(C, 1) == 0);
assert(size(C, 2) == 60);

% Vertical concat Mx0 with different row counts
A = zeros(10, 0);
B = zeros(20, 0);
C = [A; B];
assert(size(C, 1) == 30);
assert(size(C, 2) == 0);

% Mixed: empty and non-empty should still error if columns don't match
% (can't concat 30x0 with 30x5 vertically since cols differ)

% Multiplication of result should work: Mx0 * 0xN = MxN zeros
A = zeros(3, 0);
B = zeros(0, 4);
C = A * B;
assert(isequal(size(C), [3, 4]));
assert(all(all(C == 0)));

% Vertcat then multiply
A = zeros(3, 0);
B = zeros(3, 0);
C = [A; B];
D = zeros(0, 5);
E = C * D;
assert(isequal(size(E), [6, 5]));
assert(all(all(E == 0)));

fprintf('SUCCESS\n');
