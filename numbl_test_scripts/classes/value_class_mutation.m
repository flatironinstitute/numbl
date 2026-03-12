% Test that assignments create independent copies (value semantics)
% In MATLAB, assigning an object/struct/array to a new variable creates a copy.
% Modifying the copy should not affect the original.

% Test with struct
s = struct('x', 5, 'y', [1, 2, 3]);
s2 = s;
s2.x = 10;
assert(s.x == 5, 'struct should not be mutated when copy is modified');

% Test with array
a = [1, 2, 3];
b = a;
b(1) = 99;
assert(a(1) == 1, 'original array should not be mutated when copy is modified');

% Test that function calls don't mutate arguments
function modify_struct(s)
    s.x = 999;
end

s3 = struct('x', 5);
modify_struct(s3);
assert(s3.x == 5, 'struct passed to function should not be mutated');

disp('SUCCESS');
