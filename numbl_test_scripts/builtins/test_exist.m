% Test exist builtin

% Test 1: exist with 'builtin' type
assert(exist('sin', 'builtin') == 5);
assert(exist('cos', 'builtin') == 5);
assert(exist('disp', 'builtin') == 5);
assert(exist('OCTAVE_VERSION', 'builtin') == 0);
assert(exist('nonexistent_xyz', 'builtin') == 0);

% Test 2: exist with 'var' type
x = 42;
assert(exist('x', 'var') == 1);
assert(exist('nonexistent_xyz', 'var') == 0);

disp('SUCCESS');
