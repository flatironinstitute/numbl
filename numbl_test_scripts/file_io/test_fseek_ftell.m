% fseek origins, ftell, frewind, and the EOF flag for fread and fscanf.

f = [tempname '.txt'];
fid = fopen(f, 'w'); fprintf(fid, 'abc\ndef\n'); fclose(fid);

fid = fopen(f, 'r');
% --- Origins as strings and as -1/0/1 -------------------------------------
assert(fseek(fid, 2, 'bof') == 0); assert(ftell(fid) == 2);
assert(fseek(fid, 1, 'cof') == 0); assert(ftell(fid) == 3);
assert(fseek(fid, -1, 'eof') == 0); assert(ftell(fid) == 7);
assert(fseek(fid, 2, -1) == 0); assert(ftell(fid) == 2);
assert(fseek(fid, 1, 0) == 0); assert(ftell(fid) == 3);
assert(fseek(fid, -1, 1) == 0); assert(ftell(fid) == 7);
assert(fseek(fid, 0, 'cof') == 0); assert(ftell(fid) == 7);
threw = false;
try
    fseek(fid, 4);
catch
    threw = true;
end
assert(threw, 'fseek requires an origin');

% --- Seeking to the end does not set EOF; reading there does ---------------
assert(fseek(fid, 0, 'eof') == 0);
assert(ftell(fid) == 8);
assert(feof(fid) == 0);
assert(isequal(fgetl(fid), -1));
assert(feof(fid) == 1);

% --- fseek clears EOF; fgets after a seek ---------------------------------
fseek(fid, -3, 'eof');
assert(feof(fid) == 0);
assert(ftell(fid) == 5);
assert(isequal(double(fgets(fid)), [101 102 10]));

% --- Seeking before the start or past the end fails -------------------
assert(fseek(fid, -100, 'bof') == -1);
assert(ftell(fid) == 8);
assert(fseek(fid, 100, 'bof') == -1);
assert(ftell(fid) == 8);

% --- fread sets EOF only when it asks for more than remains ---------------
frewind(fid);
assert(ftell(fid) == 0);
x = fread(fid, 8);
assert(isequal(size(x), [8 1]));
assert(feof(fid) == 0);
x = fread(fid, 1);
assert(isempty(x));
assert(feof(fid) == 1);
frewind(fid);
x = fread(fid);
assert(numel(x) == 8);
assert(feof(fid) == 1);
x = fread(fid);
assert(isempty(x));
fclose(fid);

% --- fscanf -------------------------------------------------------------------
fid = fopen(f, 'w'); fprintf(fid, '1 2 3\n'); fclose(fid);
fid = fopen(f, 'r');
x = fscanf(fid, '%d', 2);
assert(isequal(x, [1; 2]));
assert(feof(fid) == 0);
assert(ftell(fid) == 3);
x = fscanf(fid, '%d');
assert(isequal(x, 3));
assert(feof(fid) == 1);
assert(ftell(fid) == 6);
fclose(fid);

% --- Initial position for append modes ------------------------------------------
fid = fopen(f, 'a');
assert(ftell(fid) == 6);
fclose(fid);
fid = fopen(f, 'a+');
assert(ftell(fid) == 0);
fclose(fid);

delete(f);
disp('SUCCESS');
