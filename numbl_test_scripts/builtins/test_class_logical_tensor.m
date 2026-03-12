% Test that class() returns 'logical' for logical tensors

% Comparison result should be logical
x = [1 2 3] > 0;
assert(strcmp(class(x), 'logical'));

% Element-wise equality
y = [1 2 3] == [1 2 4];
assert(strcmp(class(y), 'logical'));

% true/false arrays
t = true(3, 3);
assert(strcmp(class(t), 'logical'));

f = false(2, 4);
assert(strcmp(class(f), 'logical'));

% logical() cast
z = logical([1 0 1 0]);
assert(strcmp(class(z), 'logical'));

% Scalar logical should still work
assert(strcmp(class(true), 'logical'));
assert(strcmp(class(false), 'logical'));

% islogical should agree with class
assert(islogical(x));
assert(islogical(t));
assert(islogical(z));

disp('SUCCESS');
