% Test sscanf with count output

% Basic two-output form
[vals, count] = sscanf('123 456 789', '%d');
assert(count == 3);
assert(isequal(vals, [123; 456; 789]));

% Single value
[vals, count] = sscanf('42', '%d');
assert(count == 1);
assert(vals == 42);

% Float format
[vals, count] = sscanf('1.5 2.5 3.5', '%f');
assert(count == 3);
assert(abs(vals(2) - 2.5) < 1e-10);

% Partial parse (format doesn't match all)
[vals, count] = sscanf('123 abc 456', '%d');
assert(count == 1);
assert(vals == 123);

% With max count limit
[vals, count] = sscanf('1 2 3 4 5', '%d', 3);
assert(count == 3);
assert(isequal(vals, [1; 2; 3]));

disp('SUCCESS');
