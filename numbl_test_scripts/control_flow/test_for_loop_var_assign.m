% Test: Assigning to for-loop variable should not affect iteration
% In MATLAB, the for loop pre-computes the iteration range, so modifying
% the loop variable inside the body does not change the iteration sequence.

% Test 1: Basic case - assign constant to loop variable
vals = [];
for i = 1:5
    vals(end+1) = i;
    i = 100;
end
assert(isequal(vals, [1, 2, 3, 4, 5]), 'assigning constant to loop var affected iteration');

% Test 2: Assign computed value to loop variable
vals = [];
for i = 1:4
    vals(end+1) = i;
    i = i * 10;
end
assert(isequal(vals, [1, 2, 3, 4]), 'assigning computed value to loop var affected iteration');

% Test 3: Assign zero to loop variable
vals = [];
for i = 1:3
    vals(end+1) = i;
    i = 0;
end
assert(isequal(vals, [1, 2, 3]), 'assigning zero to loop var affected iteration');

% Test 4: With step
vals = [];
for i = 2:2:10
    vals(end+1) = i;
    i = 1000;
end
assert(isequal(vals, [2, 4, 6, 8, 10]), 'assigning to loop var with step affected iteration');

% Test 5: Negative step
vals = [];
for i = 5:-1:1
    vals(end+1) = i;
    i = -999;
end
assert(isequal(vals, [5, 4, 3, 2, 1]), 'assigning to loop var with negative step affected iteration');

% Test 6: Loop variable value after assignment is visible within same iteration
for i = 1:3
    i = i * 10;
    if i == 10
        % First iteration: i was 1, now 10
        assert(i == 10, 'loop var not reassigned within iteration');
    end
end

disp('SUCCESS');
