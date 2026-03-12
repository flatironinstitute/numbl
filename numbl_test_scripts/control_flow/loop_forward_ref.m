% Test that variables assigned later in a loop body are accessible in
% subsequent iterations (MATLAB workspace semantics).

% For loop
x = 0;
for c = 1:3
    if c == 1
        x = x + 1;
    else
        x = x + y;
    end
    y = 5;
end
assert(x == 11);

% While loop
x2 = 0;
c2 = 1;
while c2 <= 3
    if c2 == 1
        x2 = x2 + 1;
    else
        x2 = x2 + z;
    end
    z = 5;
    c2 = c2 + 1;
end
assert(x2 == 11);

disp('SUCCESS');
