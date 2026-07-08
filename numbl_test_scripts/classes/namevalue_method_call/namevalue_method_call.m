% Test Name=Value syntax in method-call (dot) syntax: obj.method(x, name=value)

obj = NvOpts_(2);

% method call with name=value
r = obj.scale(10, factor=3);
assert(r == 2 * 10 * 3);

% equivalent classic name-value pair
r2 = obj.scale(10, 'factor', 4);
assert(r2 == 2 * 10 * 4);

% default when omitted
r3 = obj.scale(10);
assert(r3 == 2 * 10 * 1);

% static method call with name=value
r4 = NvOpts_.combine(5, offset=2);
assert(r4 == 7);

disp('SUCCESS')
