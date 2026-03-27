% Test containers.Map shim

% Create Map with cell keys and numeric values
keySet = {'Jan','Feb','Mar','Apr'};
valueSet = [327.2 368.2 197.6 178.4];
M = containers.Map(keySet, valueSet);
assert(M.Count == 4);
assert(strcmp(M.KeyType, 'char'));
assert(strcmp(M.ValueType, 'double'));
assert(M('Mar') == 197.6);

% Modify value
M('Jan') = 100;
assert(M('Jan') == 100);

% Add new entry
M('May') = 200;
assert(M.Count == 5);
assert(M('May') == 200);

% isKey
assert(isKey(M, 'Feb'));
assert(~isKey(M, 'Jun'));

% keys and values
k = keys(M);
assert(iscell(k));
assert(length(k) == 5);

v = values(M);
assert(iscell(v));
assert(length(v) == 5);

% remove
remove(M, 'Apr');
assert(M.Count == 4);
assert(~isKey(M, 'Apr'));

% length and size
assert(length(M) == 4);

% Empty map with type spec
M2 = containers.Map('KeyType', 'char', 'ValueType', 'double');
assert(M2.Count == 0);
M2('x') = 42;
assert(M2('x') == 42);
assert(M2.Count == 1);

% Numeric keys
ids = [437 1089 2362];
names = {'Lee, N.','Jones, R.','Sanchez, C.'};
M3 = containers.Map(ids, names);
assert(M3.Count == 3);
assert(strcmp(M3.KeyType, 'double'));
assert(strcmp(M3.ValueType, 'char'));

% Empty map (no args)
M4 = containers.Map;
assert(M4.Count == 0);

% Non-uniform values
keySet2 = {'Li','Jones','Sanchez'};
valueSet2 = {[5.8 7.35], [27 3.92], 'test'};
M5 = containers.Map(keySet2, valueSet2, 'UniformValues', false);
assert(M5.Count == 3);
assert(strcmp(M5.ValueType, 'any'));

disp('SUCCESS');
