% Test rmdir builtin

% Create a temp directory then remove it
mkdir('tmp_test_rmdir_dir');
rmdir('tmp_test_rmdir_dir');
assert(exist('tmp_test_rmdir_dir', 'dir') == 0, 'rmdir failed to remove directory');

% Test with status output
mkdir('tmp_test_rmdir_dir2');
[status, msg, msgid] = rmdir('tmp_test_rmdir_dir2');
assert(status == 1, 'rmdir status should be 1 on success');

% Test recursive removal
mkdir('tmp_test_rmdir_nested/sub1/sub2');
rmdir('tmp_test_rmdir_nested', 's');
assert(exist('tmp_test_rmdir_nested', 'dir') == 0, 'rmdir recursive failed');

% Test failure case with output capture
[status, msg, msgid] = rmdir('nonexistent_dir_xyz_123');
assert(status == 0, 'rmdir should return 0 for nonexistent directory');

disp('SUCCESS');
