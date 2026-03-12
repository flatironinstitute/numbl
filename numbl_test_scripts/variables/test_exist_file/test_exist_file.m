% Test exist() with 'file' option
% my_workspace_helper.m is a workspace function in the same directory

% Workspace function file should return 2
assert(exist('my_workspace_helper', 'file') == 2, 'workspace function should return 2 for file');

% Non-existent file should return 0
assert(exist('nonexistent_file_xyz', 'file') == 0, 'nonexistent file should return 0');

% Built-in is not a file
assert(exist('sin', 'file') == 0, 'builtin is not a file');

disp('SUCCESS');
