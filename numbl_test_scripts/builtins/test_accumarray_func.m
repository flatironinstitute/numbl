% Test accumarray builtin

% Basic sum accumulation
subs = [1; 1; 2; 2; 3];
vals = [10; 20; 30; 40; 50];
result = accumarray(subs, vals);
assert(isequal(result, [30; 70; 50]));

% With function handle
result2 = accumarray(subs, vals, [], @max);
assert(isequal(result2, [20; 40; 50]));

% With gaps (fill with 0)
subs2 = [1; 3; 3];
vals2 = [5; 10; 20];
result3 = accumarray(subs2, vals2);
assert(isequal(result3, [5; 0; 30]));

% Explicit size
result4 = accumarray(subs2, vals2, [5, 1]);
assert(isequal(result4, [5; 0; 30; 0; 0]));

% Scalar val (replicated)
result5 = accumarray([1; 1; 2; 3; 3], 1);
assert(isequal(result5, [2; 1; 2]));

disp('SUCCESS');
