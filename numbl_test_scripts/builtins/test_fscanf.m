% Test fscanf: formatted matrix reads and text/position coherence with fgetl

d = tempdir;
fname = fullfile(d, 'numbl_fscanf_test.txt');
fid = fopen(fname, 'w');
fprintf(fid, 'header\n');
fprintf(fid, '3\n');
fprintf(fid, '1 10 100\n2 20 200\n3 30 300\n');
fprintf(fid, 'footer\n');
fclose(fid);

% ftell reflects what fgetl consumed; fseek rewinds
fid = fopen(fname, 'r');
h = fgetl(fid);
assert(strcmp(h, 'header'));
assert(ftell(fid) == 7);
fseek(fid, 0, 'bof');
assert(strcmp(fgetl(fid), 'header'));

% matrix read: fills column-major, so transpose recovers the rows
n = sscanf(fgetl(fid), '%d');
assert(n == 3);
data = fscanf(fid, '%f', [3, n])';
assert(isequal(size(data), [3 3]));
assert(isequal(data, [1 10 100; 2 20 200; 3 30 300]));

% fscanf stops right after the last matched value; fgetl picks up from there
rest = fgetl(fid);
assert(isempty(rest));
foot = fgetl(fid);
assert(strcmp(foot, 'footer'));
fclose(fid);

% scalar read with count output
fid = fopen(fname, 'r');
fgetl(fid);
[val, cnt] = fscanf(fid, '%d', 1);
assert(val == 3);
assert(cnt == 1);
fclose(fid);

delete(fname);
disp('SUCCESS')
