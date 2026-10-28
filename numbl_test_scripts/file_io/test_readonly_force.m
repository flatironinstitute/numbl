% Read-only files: fopen refuses to write them, and copyfile/movefile
% replace them only with the 'f' option.

d = tempname;
mkdir(d);
src = fullfile(d, 'src.txt');
dst = fullfile(d, 'dst.txt');
fid = fopen(src, 'w'); fprintf(fid, 'new'); fclose(fid);
fid = fopen(dst, 'w'); fprintf(fid, 'old'); fclose(fid);

% --- fileattrib can clear the write bit ---------------------------------------
[status, msg] = fileattrib(dst, '-w');
assert(status == 1 && isempty(msg));
[~, a] = fileattrib(dst);
assert(a.UserWrite == 0);

% --- fopen for writing fails; reading still works ---------------------------
[fid, msg] = fopen(dst, 'w');
assert(fid == -1 && strcmp(msg, 'Permission denied'));
[fid, msg] = fopen(dst, 'a');
assert(fid == -1 && strcmp(msg, 'Permission denied'));
[fid, msg] = fopen(dst, 'r+');
assert(fid == -1 && ~isempty(msg));
fid = fopen(dst, 'r');
assert(fid >= 3);
assert(strcmp(fgetl(fid), 'old'));
fclose(fid);

% --- copyfile needs 'f' to replace a read-only file -------------------------
status = copyfile(src, dst);
assert(status == 0);
assert(strcmp(fileread(dst), 'old'));
status = copyfile(src, dst, 'f');
assert(status == 1);
assert(strcmp(fileread(dst), 'new'));
[~, a] = fileattrib(dst);
assert(a.UserWrite == 1);

% --- movefile likewise ----------------------------------------------------------
fileattrib(dst, '-w');
src2 = fullfile(d, 'src2.txt');
fid = fopen(src2, 'w'); fprintf(fid, 'moved'); fclose(fid);
status = movefile(src2, dst);
assert(status == 0);
assert(strcmp(fileread(dst), 'new'));
assert(isfile(src2));
status = movefile(src2, dst, 'f');
assert(status == 1);
assert(strcmp(fileread(dst), 'moved'));
assert(~isfile(src2));
[~, a] = fileattrib(dst);
assert(a.UserWrite == 1);

% --- +w restores write access ----------------------------------------------------
fileattrib(dst, '-w');
fileattrib(dst, '+w');
fid = fopen(dst, 'a');
assert(fid >= 3);
fclose(fid);

rmdir(d, 's');
disp('SUCCESS');
