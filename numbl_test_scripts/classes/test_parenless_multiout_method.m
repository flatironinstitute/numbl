% [a, b] = obj.method; (no parens) must bind all requested outputs.

m = PulseqPatterns_();

[x, y] = m.two;
assert(x == 1);
assert(y == 2);

% With parens, unchanged
[x2, y2] = m.two();
assert(x2 == 1 && y2 == 2);

% Single output forms
z = m.two;
assert(z == 1);
z2 = m.two();
assert(z2 == 1);

disp('SUCCESS');
