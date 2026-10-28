% Writes through one fid are visible right away to other fids on the same
% file and to whole-file functions (fileread, dir, copyfile).

f = [tempname '.txt'];

% --- fileread and a second fid see unclosed writes ------------------------------
fid = fopen(f, 'w');
fprintf(fid, 'abc\ndef\n');
assert(strcmp(fileread(f), sprintf('abc\ndef\n')));
fid2 = fopen(f, 'r');
assert(strcmp(fgetl(fid2), 'abc'));
assert(feof(fid2) == 0);
fclose(fid);
assert(strcmp(fgetl(fid2), 'def'));
assert(isequal(fgetl(fid2), -1));
fclose(fid2);

% --- A reader that reached EOF stays there until it seeks ----------------
fid = fopen(f, 'w');
fid2 = fopen(f, 'r');
fprintf(fid, 'line1\n');
assert(strcmp(fgetl(fid2), 'line1'));
assert(feof(fid2) == 1);
fprintf(fid, 'line2\n');
assert(isequal(fgetl(fid2), -1));
fseek(fid2, 0, 'cof');
assert(strcmp(fgetl(fid2), 'line2'));
fclose(fid);
fclose(fid2);

% --- The EOF flag also survives writes on the same fid ---------------------------
fid = fopen(f, 'w+');
fprintf(fid, 'ab\n');
frewind(fid);
assert(strcmp(fgetl(fid), 'ab'));
assert(feof(fid) == 1);
fprintf(fid, 'cd\n');
assert(feof(fid) == 1);
assert(ftell(fid) == 6);
assert(isequal(fgetl(fid), -1));
assert(isempty(fread(fid)));
fseek(fid, 3, 'bof');
assert(strcmp(fgetl(fid), 'cd'));
fclose(fid);

% --- dir and copyfile see unclosed writes ------------------------------------------
fid = fopen(f, 'w');
fprintf(fid, 'hello');
d = dir(f);
assert(d.bytes == 5);
g = [tempname '.txt'];
copyfile(f, g);
assert(strcmp(fileread(g), 'hello'));
delete(g);
fclose(fid);

% --- Opening with 'w' truncates for every fid on the file -------------------------
fid = fopen(f, 'r');
fid2 = fopen(f, 'w');
assert(isequal(fgetl(fid), -1));
fprintf(fid2, 'new\n');
frewind(fid);
assert(strcmp(fgetl(fid), 'new'));
fclose(fid);
fclose(fid2);

delete(f);
disp('SUCCESS');
