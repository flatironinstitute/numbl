% fprintf(2, ...) writes to standard error and returns the byte count.

n = fprintf(2, 'stderr line %d\n', 7);
assert(n == 14);
n = fprintf(1, '');
assert(n == 0);

disp('SUCCESS');
