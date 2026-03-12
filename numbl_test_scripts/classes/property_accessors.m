% Test property get and set methods

% Test 1: Dependent property with getter
t = TriArea_;
t.Base = 3;
t.Height = 4;
assert(t.Area == 6, 'Getter should compute area');

% Test 2: Dependent property with default values
t2 = TriArea_;
assert(t2.Area == 0.5, 'Default area should be 0.5 (Base=1, Height=1)');

% Test 3: Setter with validation (clamps to 100)
c = Clamped_;
c.Value = 150;
assert(c.Value == 100, 'Setter should clamp to 100');

% Test 4: Setter allows valid value
c2 = Clamped_;
c2.Value = 42;
assert(c2.Value == 42, 'Setter should allow valid values');

disp('SUCCESS');
