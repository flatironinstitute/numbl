% fgetl/fgets line splitting and MATLAB's feof rules.

f = [tempname '.txt'];

% --- feof becomes true when a line read reaches the end -------------------
fid = fopen(f, 'w'); fprintf(fid, 'abc\ndef\n'); fclose(fid);
fid = fopen(f, 'r');
assert(feof(fid) == 0);
assert(strcmp(fgetl(fid), 'abc'));
assert(feof(fid) == 0);
assert(strcmp(fgetl(fid), 'def'));
assert(feof(fid) == 1);
l = fgetl(fid);
assert(isequal(l, -1) && isa(l, 'double'));
assert(isequal(fgets(fid), -1));
frewind(fid);
assert(feof(fid) == 0);
assert(isequal(fgets(fid), sprintf('abc\n')));
fclose(fid);

% --- The usual read loop sees every line exactly once ----------------------
fid = fopen(f, 'r');
lines = {};
while ~feof(fid)
    lines{end+1} = fgetl(fid); %#ok<SAGROW>
end
fclose(fid);
assert(isequal(lines, {'abc', 'def'}));

% --- Last line without a newline ----------------------------------------------
fid = fopen(f, 'w'); fprintf(fid, 'x\ny'); fclose(fid);
fid = fopen(f, 'r');
assert(strcmp(fgetl(fid), 'x'));
assert(feof(fid) == 0);
assert(strcmp(fgetl(fid), 'y'));
assert(feof(fid) == 1);
assert(isequal(fgetl(fid), -1));
fclose(fid);

% --- Empty file and empty lines ---------------------------------------------
fid = fopen(f, 'w'); fclose(fid);
fid = fopen(f, 'r');
assert(feof(fid) == 0);
assert(isequal(fgetl(fid), -1));
assert(feof(fid) == 1);
fclose(fid);
fid = fopen(f, 'w'); fprintf(fid, '\n\nx'); fclose(fid);
fid = fopen(f, 'r');
l = fgetl(fid);
assert(ischar(l) && isempty(l));
assert(isequal(double(fgets(fid)), 10));
assert(strcmp(fgetl(fid), 'x'));
fclose(fid);

% --- CRLF and lone CR end lines; fgets keeps the terminator ---------------
fid = fopen(f, 'w'); fwrite(fid, [97 13 10 98 13 10]); fclose(fid);
fid = fopen(f, 'r');
assert(isequal(double(fgetl(fid)), 97));
assert(isequal(double(fgets(fid)), [98 13 10]));
fclose(fid);
fid = fopen(f, 'w'); fwrite(fid, [97 13 98 10 99 13]); fclose(fid);
fid = fopen(f, 'r');
assert(strcmp(fgetl(fid), 'a'));
assert(strcmp(fgetl(fid), 'b'));
assert(strcmp(fgetl(fid), 'c'));
assert(feof(fid) == 1);
frewind(fid);
assert(isequal(double(fgets(fid)), [97 13]));
fclose(fid);

% --- fgets with a character limit --------------------------------------------
fid = fopen(f, 'w'); fprintf(fid, 'abcdef\ngh\n'); fclose(fid);
fid = fopen(f, 'r');
assert(strcmp(fgets(fid, 3), 'abc'));
assert(ftell(fid) == 3);
assert(isequal(double(fgets(fid, 10)), [100 101 102 10]));
assert(strcmp(fgetl(fid), 'gh'));
fclose(fid);

% --- Reading from a write-only fid gives empty results -------------------
fid = fopen(f, 'w');
fprintf(fid, 'zz\n');
l = fgetl(fid);
assert(ischar(l) && isempty(l));
x = fread(fid);
assert(isempty(x));
fclose(fid);

delete(f);
disp('SUCCESS');
