% Test setappdata / getappdata / rmappdata / isappdata

% Basic set and get with handle 0
setappdata(0, 'myval', 42);
assert(getappdata(0, 'myval') == 42);

% Overwrite
setappdata(0, 'myval', 99);
assert(getappdata(0, 'myval') == 99);

% Multiple keys
setappdata(0, 'a', 1);
setappdata(0, 'b', 2);
assert(getappdata(0, 'a') == 1);
assert(getappdata(0, 'b') == 2);

% getappdata with missing key returns []
v = getappdata(0, 'nonexistent');
assert(isempty(v));

% getappdata(obj) returns struct with all values
s = getappdata(0);
assert(isstruct(s));
assert(s.a == 1);
assert(s.b == 2);

% isappdata
assert(isappdata(0, 'a'));
assert(~isappdata(0, 'nonexistent'));

% rmappdata
rmappdata(0, 'a');
assert(~isappdata(0, 'a'));
assert(isappdata(0, 'b'));

% Different handles are independent
setappdata(1, 'x', 10);
assert(getappdata(1, 'x') == 10);
assert(~isappdata(0, 'x'));

% Store non-scalar values
setappdata(0, 'vec', [1 2 3]);
v = getappdata(0, 'vec');
assert(isequal(v, [1 2 3]));

disp('SUCCESS');
