% Test delete (file deletion)

% Delete a single file
f = tempname;
fid = fopen(f, 'w');
fprintf(fid, 'hello');
fclose(fid);
assert(exist(f, 'file') == 2);
delete(f);
assert(exist(f, 'file') == 0);

% Delete with glob pattern
d = tempdir;
f1 = fullfile(d, 'numbl_deltest_a.tmp');
f2 = fullfile(d, 'numbl_deltest_b.tmp');
fid = fopen(f1, 'w'); fprintf(fid, 'a'); fclose(fid);
fid = fopen(f2, 'w'); fprintf(fid, 'b'); fclose(fid);
assert(exist(f1, 'file') == 2);
assert(exist(f2, 'file') == 2);
delete(fullfile(d, 'numbl_deltest_*.tmp'));
assert(exist(f1, 'file') == 0);
assert(exist(f2, 'file') == 0);

% Delete multiple files as separate arguments
f3 = tempname;
f4 = tempname;
fid = fopen(f3, 'w'); fprintf(fid, 'c'); fclose(fid);
fid = fopen(f4, 'w'); fprintf(fid, 'd'); fclose(fid);
delete(f3, f4);
assert(exist(f3, 'file') == 0);
assert(exist(f4, 'file') == 0);

disp('SUCCESS');
