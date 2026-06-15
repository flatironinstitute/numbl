% figure(Name,Value) creation, ishandle, and ver.

% Name-value figure creation must not crash trying to coerce 'Visible' to a
% number; it returns a (numeric) figure handle.
fig = figure('Visible', 'off');
assert(ishandle(fig), 'figure handle is a valid handle');

f2 = figure('Color', 'white', 'Name', 'test', 'NumberTitle', 'off');
assert(ishandle(f2), 'second name-value figure');

% Numeric figure selection still works.
f3 = figure(7);
assert(f3 == 7, 'figure(n) returns n');

% ishandle semantics.
assert(ishandle(0), 'root handle');
assert(~ishandle(-1), 'negative is not a handle');
assert(~ishandle('x'), 'char is not a handle');
assert(isequal(ishandle([0 1 -1]), logical([1 1 0])), 'elementwise ishandle');

% ver returns a struct with a Version field.
v = ver('MATLAB');
assert(ischar(v.Version), 'ver Version is char');

disp('SUCCESS');
