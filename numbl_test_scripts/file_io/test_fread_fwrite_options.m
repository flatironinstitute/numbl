% fread/fwrite skip arguments, integer saturation, and the byte order
% given to fopen.

f = [tempname '.bin'];

% --- fwrite skips before each value, zero-filling past the end ------------
fid = fopen(f, 'w');
assert(fwrite(fid, [1 2 3], 'uint8', 2) == 3);
fclose(fid);
fid = fopen(f, 'r'); x = fread(fid)'; fclose(fid);
assert(isequal(x, [0 0 1 0 0 2 0 0 3]));

% --- Skipping inside existing data leaves it untouched ------------------------
fid = fopen(f, 'w'); fwrite(fid, 9:-1:1); fclose(fid);
fid = fopen(f, 'r+'); fwrite(fid, [70 71], 'uint8', 2); fclose(fid);
fid = fopen(f, 'r'); x = fread(fid)'; fclose(fid);
assert(isequal(x, [9 8 70 6 5 71 3 2 1]));

% --- fread skips after each value (or each N values with N*type) ------------
fid = fopen(f, 'r');
x = fread(fid, Inf, 'uint8', 2)';
assert(isequal(x, [9 6 3]));
assert(feof(fid) == 1);
fclose(fid);
fid = fopen(f, 'r');
x = fread(fid, 3, 'uint8', 1)';
assert(isequal(x, [9 70 5]));
assert(ftell(fid) == 6);
fclose(fid);
fid = fopen(f, 'r');
x = fread(fid, Inf, '2*uint8', 1)';
assert(isequal(x, [9 8 6 5 3 2]));
fclose(fid);

% --- Integer precisions round and saturate ---------------------------------
fid = fopen(f, 'w');
assert(fwrite(fid, [2.6 -5 300 NaN Inf], 'uint8') == 5);
fclose(fid);
fid = fopen(f, 'r'); x = fread(fid)'; fclose(fid);
assert(isequal(x, [3 0 255 0 255]));
fid = fopen(f, 'w'); fwrite(fid, [233 -200 1.5], 'int8'); fclose(fid);
fid = fopen(f, 'r'); x = fread(fid, Inf, 'int8')'; fclose(fid);
assert(isequal(x, [127 -128 2]));

% --- Byte order from fopen applies unless overridden ----------------------
fid = fopen(f, 'w', 'b'); fwrite(fid, 258, 'uint16'); fclose(fid);
fid = fopen(f, 'r'); x = fread(fid)'; fclose(fid);
assert(isequal(x, [1 2]));
fid = fopen(f, 'r', 'b'); x = fread(fid, 1, 'uint16'); fclose(fid);
assert(x == 258);
fid = fopen(f, 'r', 'b'); x = fread(fid, 1, 'uint16', 0, 'l'); fclose(fid);
assert(x == 513);
fid = fopen(f, 'r'); x = fread(fid, 1, 'uint16'); fclose(fid);
assert(x == 513);

delete(f);
disp('SUCCESS');
