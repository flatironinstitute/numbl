% Test advanced struct functionality

%% Dynamic field names with s.(name)
s.x = 10;
s.y = 20;
s.z = 30;
fname = 'y';
assert(s.(fname) == 20);

%% Dynamic field assignment
s2 = struct();
fields = {'a', 'b', 'c'};
for i = 1:length(fields)
    s2.(fields{i}) = i * 10;
end
assert(s2.a == 10);
assert(s2.b == 20);
assert(s2.c == 30);

%% Struct containing function handle
ops.add = @(a, b) a + b;
ops.mul = @(a, b) a * b;
assert(ops.add(3, 4) == 7);
assert(ops.mul(3, 4) == 12);

%% Iterating over struct fields with fieldnames
data.alpha = 1;
data.beta = 2;
data.gamma = 3;
fnames = fieldnames(data);
total = 0;
for i = 1:length(fnames)
    total = total + data.(fnames{i});
end
assert(total == 6);

%% Nested struct with dynamic access
config.db.host = 'localhost';
config.db.port = 5432;
section = 'db';
key = 'port';
assert(config.(section).(key) == 5432);

%% Struct with matrix field - indexing into it
point.coords = [3, 4, 5];
assert(point.coords(1) == 3);
assert(point.coords(2) == 4);
assert(point.coords(3) == 5);

%% Struct with cell field - indexing into it
record.tags = {'math', 'science', 'art'};
assert(strcmp(record.tags{1}, 'math'));
assert(strcmp(record.tags{3}, 'art'));

%% Copy semantics - structs are value types
a.val = 10;
b = a;
b.val = 20;
assert(a.val == 10);
assert(b.val == 20);

%% Deeply nested struct modification
tree.left.left.val = 1;
tree.left.right.val = 2;
tree.right.val = 3;
assert(tree.left.left.val == 1);
assert(tree.left.right.val == 2);
assert(tree.right.val == 3);

%% Struct field overwrite preserves other fields
rec.name = 'test';
rec.value = 42;
rec.name = 'updated';
assert(strcmp(rec.name, 'updated'));
assert(rec.value == 42);

%% fieldnames preserves insertion order
ordered = struct();
ordered.zebra = 1;
ordered.apple = 2;
ordered.mango = 3;
fn = fieldnames(ordered);
assert(strcmp(fn{1}, 'zebra'));
assert(strcmp(fn{2}, 'apple'));
assert(strcmp(fn{3}, 'mango'));

%% rmfield
s3 = struct('a', 1, 'b', 2, 'c', 3);
s3 = rmfield(s3, 'b');
assert(~isfield(s3, 'b'));
assert(isfield(s3, 'a'));
assert(isfield(s3, 'c'));
assert(length(fieldnames(s3)) == 2);

disp('SUCCESS')
