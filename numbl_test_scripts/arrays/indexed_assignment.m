% Indexed assignment to undefined variables (auto-creation)

% Basic: a(3) = 5 should create a = [0, 0, 5]
a(3) = 5;
assert(length(a) == 3)
assert(a(1) == 0)
assert(a(2) == 0)
assert(a(3) == 5)

% Assignment to first element
b(1) = 42;
assert(length(b) == 1)
assert(b(1) == 42)

% Subsequent indexed assignment should still work (growing)
c = [1, 2, 3];
c(5) = 10;
assert(length(c) == 5)
assert(c(1) == 1)
assert(c(4) == 0)
assert(c(5) == 10)

disp('SUCCESS')
