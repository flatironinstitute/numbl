% sprintf %g preserves the sign of negative zero (like MATLAB and C).

x = 0;
nz = -x;             % negative zero
assert(1 / nz == -Inf);  % confirm it really is -0

assert(strcmp(sprintf('%g', nz), '-0'));
assert(strcmp(sprintf('%.6g', nz), '-0'));
assert(strcmp(sprintf('%G', nz), '-0'));

% Positive zero is unaffected
assert(strcmp(sprintf('%g', 0), '0'));
assert(strcmp(sprintf('%.9g', 0), '0'));

% The array-formatting idiom used for event-library keys
s = sprintf('%.6g ', [0 nz 1.5]);
assert(strcmp(s, '0 -0 1.5 '));

disp('SUCCESS');
