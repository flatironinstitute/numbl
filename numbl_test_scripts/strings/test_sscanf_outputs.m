% sscanf supports the count, errmsg, and nextindex outputs.
[a, c, e, d] = sscanf('3.14abc', '%f', 1);
assert(a == 3.14, 'value');
assert(c == 1, 'count');
assert(isempty(e), 'errmsg empty on success');
assert(d == 5, 'nextindex points past the number');

[a2, c2, e2, d2] = sscanf('xyz', '%f', 1);
assert(isempty(a2), 'no value on failure');
assert(c2 == 0, 'zero count');
assert(~isempty(e2), 'errmsg set on matching failure');
assert(d2 == 1, 'nextindex unchanged');
disp('SUCCESS');
