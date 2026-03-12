obj = JitLocal(10);
data.x = 3;
data.y = 7;
op = @(x) x * 2;
result = obj.compute(op, data);
assert(result == 13, 'Expected 13');
disp('SUCCESS');
