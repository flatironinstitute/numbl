% fopen/fclose return values, fid numbering and reuse, fopen(fid) queries,
% and invalid-fid errors.

f = [tempname '.txt'];

% --- Missing files: fid -1 and an OS message ----------------------------
[fid, msg] = fopen(f, 'r');
assert(fid == -1);
assert(strcmp(msg, 'No such file or directory'));
[fid, msg] = fopen(f);
assert(fid == -1);
assert(strcmp(msg, 'No such file or directory'));
[fid, msg] = fopen(f, 'r+');
assert(fid == -1);
assert(strcmp(msg, 'No such file or directory'));
[fid, msg] = fopen(fullfile(tempname, 'a.txt'), 'w');
assert(fid == -1);
assert(strcmp(msg, 'No such file or directory'));
[fid, msg] = fopen(tempdir, 'r');
assert(fid == -1);
assert(strcmp(msg, 'Is a directory'));
[fid, msg] = fopen(tempdir, 'w');
assert(fid == -1);
assert(strcmp(msg, 'Is a directory'));

% --- Success: empty message ----------------------------------------------
[fid, msg] = fopen(f, 'w');
assert(fid >= 3);
assert(ischar(msg) && isempty(msg));
assert(fclose(fid) == 0);
[fid, msg] = fopen(f);
assert(fid >= 3 && isempty(msg));
fclose(fid);

% --- Invalid permissions throw -------------------------------------------
threw = false;
try
    fopen(f, 'q');
catch e
    threw = true;
    assert(strcmp(e.message, 'Invalid permission.'));
end
assert(threw);
threw = false;
try
    fopen(f, 'rw');
catch e
    threw = true;
    assert(contains(e.message, 'exactly one of the modes'));
end
assert(threw);

% --- Lowest free fid is reused -------------------------------------------
fclose('all');
a = fopen(f, 'r'); b = fopen(f, 'r'); c = fopen(f, 'r');
assert(isequal([a b c], [3 4 5]));
fclose(b);
d = fopen(f, 'r');
assert(d == 4);
assert(isequal(openedFiles, [3 4 5]));
assert(fclose('all') == 0);
x = openedFiles;
assert(isempty(x) && isequal(size(x), [0 0]) && isa(x, 'double'));
a = fopen(f, 'r');
assert(a == 3);
fclose(a);

% --- [filename, permission, machinefmt, encoding] = fopen(fid) ------------
fid = fopen(f, 'r');
[fn, perm, mf, enc] = fopen(fid);
assert(strcmp(fn, f));
assert(strcmp(perm, 'r'));
assert(strcmp(mf, 'ieee-le.l64'));
assert(strcmp(enc, 'UTF-8'));
fclose(fid);
% 't' is dropped from the reported permission, 'b' is kept
fid = fopen(f, 'rt'); [~, perm] = fopen(fid); assert(strcmp(perm, 'r')); fclose(fid);
fid = fopen(f, 'rb'); [~, perm] = fopen(fid); assert(strcmp(perm, 'rb')); fclose(fid);
fid = fopen(f, 'w+t'); [~, perm] = fopen(fid); assert(strcmp(perm, 'w+')); fclose(fid);
fid = fopen(f, 'r+b'); [~, perm] = fopen(fid); assert(strcmp(perm, 'r+b')); fclose(fid);
fid = fopen(f, 'r', 'b'); [~, ~, mf] = fopen(fid); assert(strcmp(mf, 'ieee-be')); fclose(fid);
fid = fopen(f, 'r', 'n', 'latin1'); [~, ~, ~, enc] = fopen(fid);
assert(strcmp(enc, 'ISO-8859-1')); fclose(fid);
% Relative names are reported as absolute paths
old = cd(tempdir);
[~, name, ext] = fileparts(f);
fid = fopen([name ext], 'r');
assert(strcmp(fopen(fid), f));
fclose(fid);
cd(old);
% Standard streams and unknown fids
[fn, perm] = fopen(1);
assert(strcmp(fn, '"stdout"') && strcmp(perm, 'a'));
[fn, perm] = fopen(0);
assert(strcmp(fn, '"stdin"') && strcmp(perm, 'r'));
[fn, perm] = fopen(99);
assert(ischar(fn) && isempty(fn) && isempty(perm));

% --- Invalid fids error ---------------------------------------------------
fid = fopen(f, 'r');
fclose(fid);
calls = {@() fclose(fid), @() fgetl(fid), @() fgets(fid), @() feof(fid), ...
         @() ftell(fid), @() fseek(fid, 0, 'bof'), @() fread(fid), ...
         @() fwrite(fid, 1), @() fprintf(fid, 'x'), @() frewind(fid)};
for i = 1:numel(calls)
    threw = false;
    try
        calls{i}();
    catch e
        threw = true;
        assert(startsWith(e.message, 'Invalid file identifier.'));
    end
    assert(threw, sprintf('call %d should throw', i));
end
% std streams are not files
assert(feof(1) == 0);
assert(ftell(1) == -1);

delete(f);
disp('SUCCESS');
