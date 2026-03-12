% Test that fields() works as an alias for fieldnames()
% In MATLAB, fields and fieldnames are interchangeable for structs

s = struct('a', 1, 'b', 2, 'c', 3);

% fields should return same result as fieldnames
f = fields(s);
fn = fieldnames(s);

assert(isequal(f, fn), 'fields and fieldnames should return the same result');
assert(length(f) == 3, 'fields should return 3 field names');

disp('SUCCESS');
