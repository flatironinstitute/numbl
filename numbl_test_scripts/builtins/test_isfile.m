% Test isfile builtin
assert(isfile('package.json') == 1);
assert(isfile('nonexistent_file_xyz_12345.txt') == 0);
assert(isfile('.') == 0);

disp('SUCCESS');
