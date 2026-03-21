% Test basic loop-level JIT: accumulation with scalar builtins
a = 0;
for i = 1:1000
    a = a + sin(i);
end
assert(abs(a - 0.8139696341) < 1e-6, 'sin accumulation wrong');

% Loop variable should persist with last iteration value
assert(i == 1000, 'loop var should be 1000');

% Multiple scalar operations
b = 0;
for j = 1:100
    b = b + cos(j) * sqrt(abs(j));
end
assert(abs(b - (-1.1150538209)) < 1e-6, 'multi-op accumulation wrong');
assert(j == 100, 'loop var j should be 100');

disp('SUCCESS')
