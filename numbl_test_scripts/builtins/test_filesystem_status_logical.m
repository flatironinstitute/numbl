% mkdir, rmdir, movefile, copyfile and fileattrib return their status as a
% logical, and isfolder/isfile return logicals.

d = tempname;
s = mkdir(d);
assert(islogical(s) && s);
[s, msg] = mkdir(d, 'sub');
assert(islogical(s) && s && ischar(msg));
assert(islogical(isfolder(d)) && isfolder(d));
assert(islogical(isfile(d)) && ~isfile(d));

f1 = fullfile(d, 'a.txt');
fid = fopen(f1, 'w');
fprintf(fid, 'x');
fclose(fid);
assert(isfile(f1));

s = copyfile(f1, fullfile(d, 'b.txt'));
assert(islogical(s) && s);
[s, msg] = movefile(fullfile(d, 'b.txt'), fullfile(d, 'c.txt'));
assert(islogical(s) && s && isempty(msg));
[s, msg] = movefile(fullfile(d, 'nope.txt'), fullfile(d, 'd.txt'));
assert(islogical(s) && ~s && ~isempty(msg));

[s, v] = fileattrib(d);
assert(islogical(s) && s);
assert(islogical(v.directory) && v.directory);
assert(islogical(v.UserRead) && v.UserRead);
assert(islogical(v.UserWrite) && islogical(v.UserExecute));
[s, v] = fileattrib(f1);
assert(s && islogical(v.directory) && ~v.directory);
s = fileattrib(fullfile(d, 'nope'));
assert(islogical(s) && ~s);
[s, msg] = fileattrib(f1, '-w');
assert(islogical(s) && s && isempty(msg));
fileattrib(f1, '+w');

[s, msg] = rmdir(fullfile(d, 'nosuchdir'));
assert(islogical(s) && ~s && ~isempty(msg));
s = rmdir(d, 's');
assert(islogical(s) && s);
assert(~isfolder(d));

disp('SUCCESS');
