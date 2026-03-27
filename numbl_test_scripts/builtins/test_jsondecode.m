% Test jsondecode builtin

% Scalar number
v = jsondecode('42');
assert(v == 42);

% Scalar string
v = jsondecode('"hello"');
assert(strcmp(v, 'hello'));

% Boolean
v = jsondecode('true');
assert(v == true);
assert(islogical(v));

v = jsondecode('false');
assert(v == false);

% Null → NaN
v = jsondecode('null');
assert(isnan(v));

% Array of numbers → column vector
v = jsondecode('[1, 2, 3]');
assert(isequal(v, [1; 2; 3]));

% Array of strings → cell array of char vectors
v = jsondecode('["one", "two", "three"]');
assert(iscell(v));
assert(length(v) == 3);
assert(strcmp(v{1}, 'one'));
assert(strcmp(v{2}, 'two'));
assert(strcmp(v{3}, 'three'));

% Array of booleans → logical array
v = jsondecode('[true, false, true]');
assert(islogical(v));
assert(isequal(v, [true; false; true]));

% JSON object → struct
v = jsondecode('{"name": "Alice", "age": 30}');
assert(isstruct(v));
assert(strcmp(v.name, 'Alice'));
assert(v.age == 30);

% Nested object
v = jsondecode('{"a": {"b": 5}}');
assert(v.a.b == 5);

% Array of objects with same fields → struct array
v = jsondecode('[{"x": 1, "y": 2}, {"x": 3, "y": 4}]');
assert(isstruct(v));
assert(v(1).x == 1);
assert(v(2).y == 4);

% Null in numeric array → NaN
v = jsondecode('[1, null, 3]');
assert(isnan(v(2)));
assert(v(1) == 1);
assert(v(3) == 3);

% Empty array
v = jsondecode('[]');
assert(isempty(v));

% Nested arrays
v = jsondecode('{"data": [10, 20, 30]}');
assert(isequal(v.data, [10; 20; 30]));

disp('SUCCESS');
