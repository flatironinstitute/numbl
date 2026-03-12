% Test logical indexing on scalar values

% false logical on scalar -> empty
r = 1.5;
result = r(false);
assert(isempty(result), 'r(false) should be empty');

% true logical on scalar -> the scalar
result2 = r(true);
assert(result2 == 1.5, 'r(true) should be 1.5');

disp('SUCCESS');
