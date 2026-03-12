% Test builtin functions on struct arrays
% In MATLAB, struct arrays are type 'struct', not a separate type

s(1).x = 10; s(1).y = 'a';
s(2).x = 20; s(2).y = 'b';
s(3).x = 30; s(3).y = 'c';

% class should return 'struct' for struct arrays
assert(strcmp(class(s), 'struct'), 'class of struct array should be struct');

% isstruct should return true for struct arrays
assert(isstruct(s), 'isstruct on struct array');

% fieldnames should work on struct arrays
f = fieldnames(s);
assert(length(f) == 2, 'fieldnames count on struct array');
assert(strcmp(f{1}, 'x'), 'fieldnames first field');
assert(strcmp(f{2}, 'y'), 'fieldnames second field');

% isfield should work on struct arrays
assert(isfield(s, 'x'), 'isfield true on struct array');
assert(isfield(s, 'y'), 'isfield y on struct array');
assert(~isfield(s, 'z'), 'isfield false on struct array');

% rmfield should work on struct arrays
s2 = rmfield(s, 'y');
f2 = fieldnames(s2);
assert(length(f2) == 1, 'rmfield removes field from struct array');
assert(s2(1).x == 10, 'rmfield preserves data in struct array');
assert(s2(2).x == 20, 'rmfield preserves data element 2');

disp('SUCCESS');
