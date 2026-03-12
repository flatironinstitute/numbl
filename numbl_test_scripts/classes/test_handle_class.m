% Test handle class semantics
c = Counter_();
assert(c.Value == 0, 'initial value should be 0');

c.increment();
assert(c.Value == 1, 'value should be 1 after increment');

c.increment(5);
assert(c.Value == 6, 'value should be 6 after increment(5)');

% Handle semantics: assignment should create a reference, not a copy
c2 = c;
c2.increment(10);
assert(c.Value == 16, 'handle semantics: c should reflect c2 mutation');
assert(c2.Value == 16, 'handle semantics: c2 should be 16');

c.reset();
assert(c.Value == 0, 'reset should set value to 0');
assert(c2.Value == 0, 'handle semantics: c2 should also be 0 after c.reset');

fprintf('SUCCESS\n');
