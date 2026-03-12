% Test that double() on logical arrays clears the logical type

% Scalar logical -> double
assert(~islogical(double(true)));
assert(~islogical(double(false)));
assert(double(true) == 1);
assert(double(false) == 0);

% Logical array -> double array
la = logical([1 0 1 1 0]);
da = double(la);
assert(isequal(da, [1 0 1 1 0]));
assert(~islogical(da));
assert(strcmp(class(da), 'double'));

% Logical matrix -> double matrix
lm = logical([1 0; 0 1]);
dm = double(lm);
assert(~islogical(dm));
assert(isequal(dm, [1 0; 0 1]));

disp('SUCCESS');
