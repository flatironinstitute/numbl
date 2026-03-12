% Test struct operations

% Basic struct creation
s = struct('name', 'Alice', 'age', 30);
assert(strcmp(s.name, 'Alice'));
assert(s.age == 30);

% Modify struct field
s.age = 31;
assert(s.age == 31);

% Add new field
s.score = 95.5;
assert(s.score == 95.5);

% Nested struct
config = struct();
config.server.host = 'localhost';
config.server.port = 8080;
assert(strcmp(config.server.host, 'localhost'));
assert(config.server.port == 8080);

% fieldnames
fields = fieldnames(s);
assert(iscell(fields));
assert(length(fields) == 3);

% isfield
assert(isfield(s, 'name'));
assert(isfield(s, 'age'));
assert(~isfield(s, 'nonexistent'));

% rmfield
s2 = struct('a', 1, 'b', 2, 'c', 3);
assert(isfield(s2, 'b'));

disp('SUCCESS')
