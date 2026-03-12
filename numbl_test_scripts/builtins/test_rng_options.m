% Test rng with various calling conventions

% Test rng() with no args returns struct with Type, Seed, State
rng(42);
s = rng();
assert(isstruct(s));
assert(isfield(s, 'Type'));
assert(isfield(s, 'Seed'));
assert(isfield(s, 'State'));

% Test rng(seed, generator) syntax
rng(123, 'twister');
a1 = rand(1, 5);
rng(123, 'twister');
a2 = rand(1, 5);
assert(all(a1 == a2));

% Test t = rng returns settings before calling rng
rng(42);
t = rng();
rand(1, 10); % advance the state
rng(t); % restore previous state
b1 = rand(1, 5);
rng(42);
b2 = rand(1, 5);
assert(all(b1 == b2));

% Test the save/restore pattern used in initializeIndexRandomly
rngprev = rng();
rng(16051821, 'twister');
x1 = randi(10, 1, 5);
rng(rngprev);
assert(all(x1 >= 1 & x1 <= 10));

% Test t = rng(seed) returns previous state
rng(42);
rand(1, 3);
t = rng(99);
assert(isstruct(t));
assert(t.Seed == 42);

disp('SUCCESS');
