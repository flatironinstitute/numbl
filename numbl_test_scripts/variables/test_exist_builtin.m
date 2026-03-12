% Test exist() with 'builtin' option

% Known built-in functions should return 5
assert(exist('sin', 'builtin') == 5, 'sin is a builtin');
assert(exist('cos', 'builtin') == 5, 'cos is a builtin');
assert(exist('zeros', 'builtin') == 5, 'zeros is a builtin');
assert(exist('disp', 'builtin') == 5, 'disp is a builtin');
assert(exist('numel', 'builtin') == 5, 'numel is a builtin');
assert(exist('size', 'builtin') == 5, 'size is a builtin');
assert(exist('length', 'builtin') == 5, 'length is a builtin');
assert(exist('isempty', 'builtin') == 5, 'isempty is a builtin');

% Non-existent names should return 0
assert(exist('nonexistent_xyz_abc', 'builtin') == 0, 'nonexistent is not a builtin');
assert(exist('my_user_func_xyz', 'builtin') == 0, 'user func is not a builtin');

disp('SUCCESS');
