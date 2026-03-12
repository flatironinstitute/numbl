% Struct operations

% Basic struct creation
s.name = 'Alice';
s.age = 30;
s.score = 95.5;

assert(strcmp(s.name, 'Alice'))
assert(s.age == 30)
assert(abs(s.score - 95.5) < 0.01)

% Modify a field
s.age = 31;
assert(s.age == 31)

% Nested struct
p.position.x = 3;
p.position.y = 4;
assert(p.position.x == 3)
assert(p.position.y == 4)

% isfield
assert(isfield(s, 'name'))
assert(~isfield(s, 'missing'))

% fieldnames - returns cell array
fields = fieldnames(s);
assert(length(fields) == 3)

disp('SUCCESS')
