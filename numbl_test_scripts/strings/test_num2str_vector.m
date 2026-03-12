% Test num2str on vectors and matrices

% Scalar (already works, included for regression)
assert(strcmp(num2str(42), '42'), 'num2str scalar');
assert(strcmp(num2str(3.14), '3.14'), 'num2str float');

% Row vector
s1 = num2str([1 2 3]);
assert(ischar(s1), 'num2str vector is char');
% Should contain all numbers separated by spaces
assert(contains(s1, '1'), 'num2str vector has 1');
assert(contains(s1, '2'), 'num2str vector has 2');
assert(contains(s1, '3'), 'num2str vector has 3');

% num2str with format on scalar
s2 = num2str(3.14159, '%0.2f');
assert(strcmp(s2, '3.14'), 'num2str format');

% Column vector should also work
s3 = num2str([10; 20; 30]);
assert(ischar(s3), 'num2str col vector is char');

% Matrix
s4 = num2str([1 2; 3 4]);
assert(ischar(s4), 'num2str matrix is char');

disp('SUCCESS');
