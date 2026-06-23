% getfield / setfield: dynamic field access and assignment on structs.

s.a = 1;
s.b = 'hello';

% getfield
assert(getfield(s, 'a') == 1, 'getfield a');
assert(strcmp(getfield(s, 'b'), 'hello'), 'getfield b');

% setfield returns a new struct with the field set
t = setfield(s, 'c', 99);
assert(t.c == 99, 'setfield new field');
assert(t.a == 1, 'setfield preserves existing');
% original is unchanged
assert(~isfield(s, 'c'), 'setfield does not mutate original');

% setfield overwrites an existing field
u = setfield(s, 'a', 42);
assert(u.a == 42, 'setfield overwrite');

% nested field-name chains
n.inner.x = 5;
assert(getfield(n, 'inner', 'x') == 5, 'nested getfield');
n2 = setfield(n, 'inner', 'x', 7);
assert(n2.inner.x == 7, 'nested setfield');

% field name supplied dynamically
fn = 'b';
assert(strcmp(getfield(s, fn), 'hello'), 'dynamic field name');

disp('SUCCESS');
