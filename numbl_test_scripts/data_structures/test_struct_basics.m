% Test basic struct functionality

%% Struct creation via dot notation
s.name = 'Alice';
s.age = 30;
s.score = 95.5;
assert(isstruct(s));

%% Field access
assert(strcmp(s.name, 'Alice'));
assert(s.age == 30);
assert(abs(s.score - 95.5) < 1e-10);

%% Modify existing field
s.age = 31;
assert(s.age == 31);

%% Add new field to existing struct
s.active = true;
assert(s.active == true);

%% isstruct on non-structs
assert(~isstruct(42));
assert(~isstruct('hello'));
assert(~isstruct([1 2 3]));
assert(~isstruct({1, 2}));

%% class() of struct
assert(strcmp(class(s), 'struct'));

%% struct() constructor with field-value pairs
s2 = struct('x', 10, 'y', 20, 'z', 30);
assert(s2.x == 10);
assert(s2.y == 20);
assert(s2.z == 30);

%% Empty struct
s3 = struct();
assert(isstruct(s3));

%% fieldnames returns cell array of field names
s4 = struct('a', 1, 'b', 2, 'c', 3);
fields = fieldnames(s4);
assert(iscell(fields));
assert(length(fields) == 3);
assert(strcmp(fields{1}, 'a'));
assert(strcmp(fields{2}, 'b'));
assert(strcmp(fields{3}, 'c'));

%% isfield
assert(isfield(s4, 'a'));
assert(isfield(s4, 'b'));
assert(isfield(s4, 'c'));
assert(~isfield(s4, 'd'));
assert(~isfield(s4, 'nonexistent'));

%% Nested struct creation
p.position.x = 3;
p.position.y = 4;
assert(p.position.x == 3);
assert(p.position.y == 4);
assert(isstruct(p.position));

%% Deeper nesting
config.server.database.host = 'localhost';
config.server.database.port = 5432;
config.server.database.name = 'mydb';
assert(strcmp(config.server.database.host, 'localhost'));
assert(config.server.database.port == 5432);
assert(strcmp(config.server.database.name, 'mydb'));

%% Struct field containing array
s5.data = [1, 2, 3, 4, 5];
s5.label = 'test';
assert(isequal(s5.data, [1, 2, 3, 4, 5]));
assert(s5.data(3) == 3);

%% Struct field containing cell
s6.items = {'apple', 'banana', 'cherry'};
assert(strcmp(s6.items{1}, 'apple'));
assert(strcmp(s6.items{3}, 'cherry'));

%% Struct field containing another struct
inner.val = 42;
outer2.inner = inner;
assert(outer2.inner.val == 42);

%% Overwriting struct field with different type
s7.x = 10;
assert(s7.x == 10);
s7.x = 'hello';
assert(strcmp(s7.x, 'hello'));

disp('SUCCESS')
