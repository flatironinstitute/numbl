% Test: For-loop variable should retain last iterated value after loop
% In MATLAB, after 'for i = 1:n', i equals n (the last value iterated).
% The loop variable should NOT be incremented past the end of the range.

% Test 1: Basic range
for i = 1:5
end
assert(i == 5, 'loop var after 1:5 should be 5');

% Test 2: Range with step
for i = 2:2:10
end
assert(i == 10, 'loop var after 2:2:10 should be 10');

% Test 3: Negative step
for i = 5:-1:1
end
assert(i == 1, 'loop var after 5:-1:1 should be 1');

% Test 4: Single iteration
for i = 1:1
end
assert(i == 1, 'loop var after 1:1 should be 1');

% Test 5: Non-integer step
for i = 0:0.5:2
end
assert(abs(i - 2) < 1e-10, 'loop var after 0:0.5:2 should be 2');

% Test 6: Loop variable used after loop in computation
total = 0;
for k = 1:4
    total = total + k;
end
assert(total == 10, 'total should be 10');
assert(k == 4, 'k should be 4 after loop');
result = total + k;
assert(result == 14, 'total + k should be 14');

% Test 7: Nested loops - each retains last value
for i = 1:3
    for j = 1:4
    end
end
assert(i == 3, 'outer loop var should be 3');
assert(j == 4, 'inner loop var should be 4');

% Test 8: Empty range - variable should not be assigned
clear_i_exists = false;
for i_empty = 1:0
    clear_i_exists = true;
end
assert(~clear_i_exists, 'empty range should not execute body');

disp('SUCCESS');
