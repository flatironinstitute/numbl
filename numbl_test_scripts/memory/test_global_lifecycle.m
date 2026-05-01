% Global tensor variables: assignment, overwrite, and access from a
% different function must keep the buffer alive.

global GBL

GBL = [10, 20, 30];
v1 = read_global();
assert(isequal(v1, [10, 20, 30]), 'first read should match');

% Overwrite the global with a different tensor — old buffer becomes
% reclaimable (env.set release on rt.$g). Reads must still work.
GBL = [1, 2, 3, 4, 5];
v2 = read_global();
assert(isequal(v2, [1, 2, 3, 4, 5]), 'second read should match overwrite');

% Mutate the global in-place via index assignment.
GBL(3) = 99;
v3 = read_global();
assert(isequal(v3, [1, 2, 99, 4, 5]), 'in-place mutation should be visible');

% A different function reads and writes the global.
write_global([7, 8, 9]);
assert(isequal(GBL, [7, 8, 9]), 'global written from another function');

% Clear and re-set repeatedly to exercise release/acquire on the same name.
for i = 1:30
  GBL = [i, i+1, i+2];
end
assert(isequal(GBL, [30, 31, 32]), 'final overwrite should match');

disp('SUCCESS')

function out = read_global()
  global GBL
  out = GBL;
end

function write_global(v)
  global GBL
  GBL = v;
end
