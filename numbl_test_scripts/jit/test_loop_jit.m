% Test JIT compilation of for and while loops

%% For loop: simple accumulation
x = 0;
n = 100;
for i = 1:n
    x = x + i;
end
assert(x == 5050, 'for loop sum failed');
assert(i == 100, 'for loop variable should retain final value');

%% For loop with step
x = 0;
for i = 1:2:10
    x = x + i;
end
assert(x == 25, 'for loop with step failed');

%% For loop: multiple outputs
a = 0;
b = 1;
n = 10;
for i = 1:n
    c = a + b;
    a = b;
    b = c;
end
assert(b == 89, 'fibonacci for loop failed');

%% While loop: simple countdown
x = 10;
count = 0;
while x > 0
    x = x - 1;
    count = count + 1;
end
assert(x == 0, 'while loop x failed');
assert(count == 10, 'while loop count failed');

%% While loop: convergence
x = 1.0;
iters = 0;
while x > 0.001
    x = x / 2;
    iters = iters + 1;
end
assert(iters == 10, 'while loop convergence iters failed');

%% For loop with if inside
total = 0;
for i = 1:20
    if i > 10
        total = total + i;
    end
end
assert(total == 155, 'for loop with if failed');

%% For loop: new variable created inside
for i = 1:5
    tmp = i * 2;
end
assert(tmp == 10, 'variable created inside for loop should be visible');

%% Nested for loops (inner should JIT even if outer doesn't fail)
total = 0;
for i = 1:5
    for j = 1:5
        total = total + i * j;
    end
end
assert(total == 225, 'nested for loops failed');

%% While loop with break
x = 0;
while true
    x = x + 1;
    if x >= 5
        break;
    end
end
assert(x == 5, 'while loop with break failed');

disp('SUCCESS');
