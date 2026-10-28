% Read/write behavior of the fopen modes 'r+', 'w+', 'a', 'a+', and text
% vs binary mode (no newline translation on this platform).

f = [tempname '.txt'];
fid = fopen(f, 'w'); fprintf(fid, 'hello world'); fclose(fid);

% --- r+ overwrites in place ---------------------------------------------------
fid = fopen(f, 'r+');
fseek(fid, 6, 'bof');
fprintf(fid, 'WORLD');
assert(ftell(fid) == 11);
frewind(fid);
assert(strcmp(fgetl(fid), 'hello WORLD'));
fclose(fid);
assert(strcmp(fileread(f), 'hello WORLD'));

% --- r+ switches from reading to writing at the current position ---------
fid = fopen(f, 'r+');
x = fread(fid, 5, '*char')';
assert(strcmp(x, 'hello'));
fwrite(fid, '_');
assert(ftell(fid) == 6);
fclose(fid);
assert(strcmp(fileread(f), 'hello_WORLD'));

% --- a writes at the end even after a seek -------------------------------------
fid = fopen(f, 'a');
fseek(fid, 0, 'bof');
assert(ftell(fid) == 0);
fprintf(fid, '!');
assert(ftell(fid) == 12);
fclose(fid);
assert(strcmp(fileread(f), 'hello_WORLD!'));

% --- a+ reads from the start and appends -------------------------------------
fid = fopen(f, 'a+');
assert(strcmp(fgetl(fid), 'hello_WORLD!'));
fprintf(fid, '?');
assert(ftell(fid) == 13);
frewind(fid);
assert(strcmp(fgetl(fid), 'hello_WORLD!?'));
fclose(fid);

% --- w+ truncates, then reads back what it wrote ------------------------------
fid = fopen(f, 'w+');
assert(isempty(fread(fid)));
frewind(fid);
fprintf(fid, 'ab\ncd\n');
frewind(fid);
assert(strcmp(fgetl(fid), 'ab'));
assert(strcmp(fgetl(fid), 'cd'));
assert(feof(fid) == 1);
fclose(fid);

% --- w truncates at open time -----------------------------------------------------
fid = fopen(f, 'w');
d = dir(f);
assert(d.bytes == 0);
fclose(fid);

% --- Writing to a read-only fid writes nothing and sets ferror --------------
fid = fopen(f, 'w'); fprintf(fid, 'zz'); fclose(fid);
fid = fopen(f, 'r');
[m, n] = ferror(fid);
assert(isempty(m) && n == 0);
assert(fprintf(fid, 'abc') == 0);
assert(fwrite(fid, [1 2 3]) == 0);
[m, n] = ferror(fid);
assert(strcmp(m, 'Bad file descriptor') && n == 9);
fclose(fid);
assert(strcmp(fileread(f), 'zz'));

% --- Text mode does not translate newlines ----------------------------------
fid = fopen(f, 'wt'); fprintf(fid, 'a\nb\n'); fclose(fid);
fid = fopen(f, 'r'); x = fread(fid)'; fclose(fid);
assert(isequal(x, [97 10 98 10]));
fid = fopen(f, 'w'); fwrite(fid, [97 13 10 98 10]); fclose(fid);
fid = fopen(f, 'rt');
assert(isequal(double(fgets(fid)), [97 13 10]));
fclose(fid);
fid = fopen(f, 'rb');
assert(isequal(fread(fid)', [97 13 10 98 10]));
fclose(fid);

% --- fprintf returns the number of bytes written ------------------------------
fid = fopen(f, 'w');
assert(fprintf(fid, '%d,', [1 2 3]) == 6);
fclose(fid);

delete(f);
disp('SUCCESS');
