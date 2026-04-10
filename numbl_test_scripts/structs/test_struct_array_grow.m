% Test struct array auto-extension via indexed field assignment

% Assigning s(2).field should grow a scalar struct to a 1x2 struct array
s = struct();
s(1).rxi = [10 20 30];
assert(length(s) == 1, 'after s(1) assignment');
s(2).rxi = [];
assert(length(s) == 2, 'after s(2) assignment');
s(3).rxi = [40 50];
assert(length(s) == 3, 'after s(3) assignment');

% Verify values
assert(isequal(s(1).rxi, [10 20 30]), 's(1).rxi');
assert(isequal(s(2).rxi, []), 's(2).rxi');
assert(isequal(s(3).rxi, [40 50]), 's(3).rxi');

% Field concat on the expanded struct array
r = [s.rxi];
assert(isequal(r, [10 20 30 40 50]), 'struct field concat after grow');

disp('SUCCESS');
