% Test exist() with 'class' option

% Workspace class should return 8
assert(exist('MyExistClass', 'class') == 8, 'workspace class should return 8');

% Non-existent class should return 0
assert(exist('nonexistent_class_xyz', 'class') == 0, 'nonexistent class should return 0');

% Built-in functions are not classes
assert(exist('sin', 'class') == 0, 'builtin function is not a class');

disp('SUCCESS');
