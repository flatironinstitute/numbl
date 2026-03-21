% Test loop JIT with nested for loops
a = 0;
for i = 1:10
    for j = 1:10
        a = a + i * j;
    end
end
assert(a == 3025, 'nested loop sum wrong');
assert(i == 10, 'outer loop var should be 10');
assert(j == 10, 'inner loop var should be 10');

% Nested with different ranges
b = 0;
for i = 1:5
    for j = i:5
        b = b + 1;
    end
end
assert(b == 15, 'triangular sum should be 15');

disp('SUCCESS')
