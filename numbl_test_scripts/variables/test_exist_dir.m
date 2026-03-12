% Test exist() with 'dir' option

% Non-existent folder should return 0
assert(exist('nonexistent_folder_xyz', 'dir') == 0, 'nonexistent folder should return 0');

% A builtin function is not a folder
assert(exist('sin', 'dir') == 0, 'builtin is not a folder');

% A workspace variable is not a folder
my_var = 42;
assert(exist('my_var', 'dir') == 0, 'variable is not a folder');

disp('SUCCESS');
