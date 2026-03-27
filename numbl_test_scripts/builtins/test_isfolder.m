% Test isfolder builtin
assert(isfolder('.') == 1);
assert(isfolder('..') == 1);
assert(isfolder('nonexistent_folder_xyz_12345') == 0);
assert(isfolder('numbl_test_scripts') == 1);

disp('SUCCESS');
