% Test num2str on matrix input
% In MATLAB, num2str(M) where M is a matrix produces a char array
% with one row per matrix row

% Basic 2x2 matrix
M = [1 2; 3 4];
s = num2str(M);
assert(size(s, 1) == 2);  % should have 2 rows

% 3x2 matrix
M2 = [10 20; 30 40; 50 60];
s2 = num2str(M2);
assert(size(s2, 1) == 3);  % should have 3 rows

% Each row should represent that row of the matrix
% Row 1 should contain '1' and '2', Row 2 should contain '3' and '4'
row1 = strtrim(s(1, :));
row2 = strtrim(s(2, :));
assert(contains(row1, '1'));
assert(contains(row1, '2'));
assert(contains(row2, '3'));
assert(contains(row2, '4'));

% Column vector
v = [1; 2; 3];
s3 = num2str(v);
assert(size(s3, 1) == 3);

% Row vector should be 1 row
v2 = [1 2 3];
s4 = num2str(v2);
assert(size(s4, 1) == 1);

disp('SUCCESS');
