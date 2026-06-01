% sprintf cycles the format string across ALL flattened arguments; the
% output must not be truncated when an argument is a multi-element vector.
% (Regression for the C-JIT format engine's no-progress guard, which used
% to break early once it crossed from one tensor argument into the next.)

assert(strcmp(sprintf('%d %d|', [1 2 3], [4 5 6]), '1 2|3 4|5 6|'));
assert(strcmp(sprintf('%d,', [1 2 3 4]), '1,2,3,4,'));
assert(strcmp(sprintf('%d-%d-%d;', [1 2], [3 4], [5 6]), '1-2-3;4-5-6;'));

% mixed scalar + vector arguments
assert(strcmp(sprintf('%d:%d ', 0, [1 2 3]), '0:1 2:3 '));

disp('SUCCESS')
