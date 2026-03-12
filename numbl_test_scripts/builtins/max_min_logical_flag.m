% Test that max/min on logical arrays preserve the logical type

% max along dim of logical matrix should be logical
M = logical([0 1; 1 0]);
r = max(M, [], 1);
assert(islogical(r), 'max along dim 1 of logical matrix should be logical');
assert(isequal(r, logical([1 1])));

c = max(M, [], 2);
assert(islogical(c), 'max along dim 2 of logical matrix should be logical');
assert(isequal(c, logical([1; 1])));

% min along dim of logical matrix should be logical
r2 = min(M, [], 1);
assert(islogical(r2), 'min along dim 1 of logical matrix should be logical');
assert(isequal(r2, logical([0 0])));

c2 = min(M, [], 2);
assert(islogical(c2), 'min along dim 2 of logical matrix should be logical');
assert(isequal(c2, logical([0; 0])));

% max of logical vector should be logical
v = logical([0 1 0 1]);
assert(islogical(max(v)), 'max of logical vector should be logical');

% min of logical vector should be logical
assert(islogical(min(v)), 'min of logical vector should be logical');

% transpose of logical should stay logical
L = logical([1; 0; 1]);
Lt = L.';
assert(islogical(Lt), 'transpose of logical column should be logical');
assert(isequal(Lt, logical([1 0 1])));

% conjugate transpose of logical should stay logical
L2 = logical([1; 0; 1]);
Lt2 = L2';
assert(islogical(Lt2), 'conjugate transpose of logical column should be logical');
assert(isequal(Lt2, logical([1 0 1])));

% combined: max along dim then transpose should stay logical
M2 = logical([0 1; 1 0; 0 1]);
result = max(M2, [], 2).';
assert(islogical(result), 'max along dim 2 then transpose should be logical');
assert(isequal(result, logical([1 1 1])));

disp('SUCCESS');
