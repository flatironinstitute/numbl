% Test that for-loop counters are not clobbered by function calls
% that themselves contain for-loops (regression test for temp variable
% scoping bug where JIT-compiled functions reused the same global temps).

function result = inner_sum(n)
    result = 0;
    for k = 1:n
        result = result + k;
    end
end

% Case 1: function call inside a for-loop
total = 0;
for i = 1:5
    total = total + inner_sum(i);
end
% inner_sum(1)=1, inner_sum(2)=3, inner_sum(3)=6, inner_sum(4)=10, inner_sum(5)=15
assert(total == 35);

% Case 2: nested for-loop with function call
results = zeros(3, 4);
for i = 1:3
    for j = 1:4
        results(i, j) = inner_sum(i + j);
    end
end
assert(results(1, 1) == inner_sum(2));   % i=1,j=1 -> inner_sum(2)=3
assert(results(3, 4) == inner_sum(7));   % i=3,j=4 -> inner_sum(7)=28
assert(results(2, 3) == inner_sum(5));   % i=2,j=3 -> inner_sum(5)=15

% Case 3: verify all iterations execute (not stuck in infinite loop)
count = 0;
for i = 1:10
    inner_sum(3);  % function with its own for-loop
    count = count + 1;
end
assert(count == 10);

disp('SUCCESS')
