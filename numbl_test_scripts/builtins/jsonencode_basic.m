% jsonencode basic coverage

s.name = 'x';
s.vals = [1 2 3];
s.flag = true;
s.nested = struct('a', 1);
s.names = {'p', 'q'};
s.empty = [];
out = jsonencode(s);
expected = '{"name":"x","vals":[1,2,3],"flag":true,"nested":{"a":1},"names":["p","q"],"empty":[]}';
assert(strcmp(out, expected), out);

assert(strcmp(jsonencode(42), '42'));
assert(strcmp(jsonencode(-3.5), '-3.5'));
assert(strcmp(jsonencode([]), '[]'));
assert(strcmp(jsonencode({1, 'a', [2 3]}), '[1,"a",[2,3]]'));
assert(strcmp(jsonencode([1 2; 3 4]), '[[1,2],[3,4]]'));
assert(strcmp(jsonencode("hello"), '"hello"'));
assert(strcmp(jsonencode(true), 'true'));
assert(strcmp(jsonencode(struct()), '{}'));

disp('SUCCESS')
