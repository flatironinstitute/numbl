% Test loop JIT with break and continue

% Break exits early, loop var holds the break iteration value
a = 0;
for i = 1:100
    if i > 10
        break
    end
    a = a + i;
end
assert(a == 55, 'break: sum should be 55');
assert(i == 11, 'break: i should be 11');

% Continue skips even numbers
b = 0;
for j = 1:10
    if mod(j, 2) == 0
        continue
    end
    b = b + j;
end
assert(b == 25, 'continue: odd sum should be 25');
assert(j == 10, 'continue: j should be 10');

disp('SUCCESS')
