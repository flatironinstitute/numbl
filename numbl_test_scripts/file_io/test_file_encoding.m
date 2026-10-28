% Text encodings for fopen, fprintf, fgetl, fread/fwrite of char data, and
% fileread.

f = [tempname '.txt'];

% --- Default encoding is UTF-8 ---------------------------------------------------
fid = fopen(f, 'w');
fprintf(fid, '%s\n', char([99 97 102 233]));
fclose(fid);
fid = fopen(f, 'r'); x = fread(fid)'; fclose(fid);
assert(isequal(x, [99 97 102 195 169 10]));
fid = fopen(f, 'r');
assert(isequal(double(fgetl(fid)), [99 97 102 233]));
assert(ftell(fid) == 6);
fclose(fid);
fid = fopen(f, 'w'); fprintf(fid, '%s', char([99 233 8364])); fclose(fid);
fid = fopen(f, 'r'); x = fread(fid)'; fclose(fid);
assert(isequal(x, [99 195 169 226 130 172]));

% --- Invalid UTF-8: one char per byte by default, U+FFFD if explicit -------
fid = fopen(f, 'w'); fwrite(fid, [99 97 102 233 10]); fclose(fid);
fid = fopen(f, 'r');
assert(isequal(double(fgetl(fid)), [99 97 102 233]));
fclose(fid);
fid = fopen(f, 'r', 'n', 'UTF-8');
assert(isequal(double(fgetl(fid)), [99 97 102 65533]));
fclose(fid);
fid = fopen(f, 'r', 'n', 'ISO-8859-1');
assert(isequal(double(fgetl(fid)), [99 97 102 233]));
fclose(fid);
fid = fopen(f, 'w'); fwrite(fid, [99 233]); fclose(fid);
assert(isequal(double(fileread(f)), [99 233]));

% --- Single-byte encodings when writing -------------------------------------------
fid = fopen(f, 'w', 'n', 'ISO-8859-1');
fprintf(fid, '%s', char([99 97 102 233]));
fclose(fid);
fid = fopen(f, 'r'); x = fread(fid)'; fclose(fid);
assert(isequal(x, [99 97 102 233]));
fid = fopen(f, 'w', 'n', 'windows-1252');
fprintf(fid, '%s', char([99 233 8364]));
fclose(fid);
fid = fopen(f, 'r'); x = fread(fid)'; fclose(fid);
assert(isequal(x, [99 233 128]));
fid = fopen(f, 'w', 'n', 'US-ASCII');
fprintf(fid, '%s', char([99 233]));
fclose(fid);
fid = fopen(f, 'r'); x = fread(fid)'; fclose(fid);
assert(isequal(x, [99 26]));

% --- fwrite encodes char data written as bytes -------------------------------
fid = fopen(f, 'w');
assert(fwrite(fid, char([99 233 8364])) == 3);
fclose(fid);
fid = fopen(f, 'r'); x = fread(fid)'; fclose(fid);
assert(isequal(x, [99 195 169 226 130 172]));
fid = fopen(f, 'w'); fwrite(fid, char([99 233]), 'uint8'); fclose(fid);
fid = fopen(f, 'r'); x = fread(fid)'; fclose(fid);
assert(isequal(x, [99 195 169]));
fid = fopen(f, 'w'); fwrite(fid, char([99 233]), 'uint16'); fclose(fid);
fid = fopen(f, 'r'); x = fread(fid)'; fclose(fid);
assert(isequal(x, [99 0 233 0]));

% --- fread with char precision decodes; uint8=>char does not --------------
fid = fopen(f, 'w'); fwrite(fid, [99 195 169 100 10]); fclose(fid);
fid = fopen(f, 'r'); x = fread(fid, '*char')'; fclose(fid);
assert(isequal(double(x), [99 233 100 10]));
fid = fopen(f, 'r');
x = fread(fid, 2, '*char')';
assert(isequal(double(x), [99 233]));
assert(ftell(fid) == 3);
fclose(fid);
fid = fopen(f, 'r'); x = fread(fid, 'char')'; fclose(fid);
assert(isequal(x, [99 233 100 10]));
fid = fopen(f, 'r'); x = fread(fid, 'uint8=>char')'; fclose(fid);
assert(isequal(double(x), [99 195 169 100 10]));
fid = fopen(f, 'r', 'n', 'latin1'); x = fread(fid, '*char')'; fclose(fid);
assert(isequal(double(x), [99 195 169 100 10]));

delete(f);
disp('SUCCESS');
