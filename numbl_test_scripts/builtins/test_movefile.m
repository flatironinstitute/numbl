% Test movefile builtin

d = tempdir;

% --- Move (rename) a file ----------------------------------------------
src = fullfile(d, 'numbl_movetest_a.tmp');
dst = fullfile(d, 'numbl_movetest_b.tmp');
fid = fopen(src, 'w'); fprintf(fid, 'hello'); fclose(fid);
movefile(src, dst);
assert(exist(src, 'file') == 0, 'source should be gone after move');
assert(exist(dst, 'file') == 2, 'destination should exist after move');
delete(dst);

% --- Status output form -------------------------------------------------
src = fullfile(d, 'numbl_movetest_c.tmp');
dst = fullfile(d, 'numbl_movetest_d.tmp');
fid = fopen(src, 'w'); fprintf(fid, 'x'); fclose(fid);
[status, msg, msgid] = movefile(src, dst);
assert(status == 1, 'movefile status should be 1 on success');
assert(exist(dst, 'file') == 2);
delete(dst);

% --- Move into an existing directory ------------------------------------
sub = fullfile(d, 'numbl_movetest_subdir');
mkdir(sub);
src = fullfile(d, 'numbl_movetest_e.tmp');
fid = fopen(src, 'w'); fprintf(fid, 'y'); fclose(fid);
movefile(src, sub);
moved = fullfile(sub, 'numbl_movetest_e.tmp');
assert(exist(moved, 'file') == 2, 'file should be moved into directory');
assert(exist(src, 'file') == 0);
delete(moved);
rmdir(sub);

% --- Force flag overwrites existing destination -------------------------
src = fullfile(d, 'numbl_movetest_f.tmp');
dst = fullfile(d, 'numbl_movetest_g.tmp');
fid = fopen(src, 'w'); fprintf(fid, 'new'); fclose(fid);
fid = fopen(dst, 'w'); fprintf(fid, 'old'); fclose(fid);
movefile(src, dst, 'f');
assert(exist(src, 'file') == 0);
assert(exist(dst, 'file') == 2);
fid = fopen(dst, 'r');
contents = fgetl(fid);
fclose(fid);
assert(strcmp(contents, 'new'), 'destination should contain new contents');
delete(dst);

% --- Failure case: nonexistent source returns status 0 ------------------
[status, msg, msgid] = movefile('nonexistent_xyz_123.tmp', 'somewhere.tmp');
assert(status == 0, 'movefile should return 0 for nonexistent source');

% --- Move (rename) a folder ---------------------------------------------
folderA = fullfile(d, 'numbl_movetest_folderA');
folderB = fullfile(d, 'numbl_movetest_folderB');
mkdir(folderA);
inner = fullfile(folderA, 'inner.tmp');
fid = fopen(inner, 'w'); fprintf(fid, 'inside'); fclose(fid);
movefile(folderA, folderB);
assert(exist(folderA, 'dir') == 0, 'source folder should be gone');
assert(exist(folderB, 'dir') == 7, 'destination folder should exist');
assert(exist(fullfile(folderB, 'inner.tmp'), 'file') == 2);
rmdir(folderB, 's');

disp('SUCCESS');
