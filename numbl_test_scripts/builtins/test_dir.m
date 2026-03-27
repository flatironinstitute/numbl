% Test dir builtin

% Setup: create a temp directory structure
tmpBase = [tempname(), '_dir_test'];
mkdir(tmpBase);
d1 = fullfile(tmpBase, 'subdir1');
mkdir(d1);
d2 = fullfile(tmpBase, 'subdir1', 'subdir2');
mkdir(d2);

% Create test files
fid = fopen(fullfile(tmpBase, 'file1.txt'), 'w');
fprintf(fid, 'hello');
fclose(fid);

fid = fopen(fullfile(tmpBase, 'file2.m'), 'w');
fprintf(fid, 'world');
fclose(fid);

fid = fopen(fullfile(d1, 'file3.txt'), 'w');
fprintf(fid, 'nested');
fclose(fid);

fid = fopen(fullfile(d2, 'file4.dat'), 'w');
fprintf(fid, 'deep');
fclose(fid);

% Test 1: dir with no output args (display mode) - just make sure it doesn't error
dir(tmpBase);

% Test 2: listing = dir(folder) returns struct array
listing = dir(tmpBase);
assert(isstruct(listing), 'dir should return a struct');
assert(isfield(listing, 'name'), 'struct should have name field');
assert(isfield(listing, 'folder'), 'struct should have folder field');
assert(isfield(listing, 'date'), 'struct should have date field');
assert(isfield(listing, 'bytes'), 'struct should have bytes field');
assert(isfield(listing, 'isdir'), 'struct should have isdir field');
assert(isfield(listing, 'datenum'), 'struct should have datenum field');

% Check expected entries: . .. file1.txt file2.m subdir1
n = length(listing);
assert(n >= 5, 'should have at least 5 entries');

foundDot = false; foundDotDot = false; foundFile1 = false;
foundFile2 = false; foundSubdir = false;
for i = 1:n
    nm = listing(i).name;
    if strcmp(nm, '.'), foundDot = true; end
    if strcmp(nm, '..'), foundDotDot = true; end
    if strcmp(nm, 'file1.txt'), foundFile1 = true; end
    if strcmp(nm, 'file2.m'), foundFile2 = true; end
    if strcmp(nm, 'subdir1'), foundSubdir = true; end
end
assert(foundDot, 'should contain .');
assert(foundDotDot, 'should contain ..');
assert(foundFile1, 'should contain file1.txt');
assert(foundFile2, 'should contain file2.m');
assert(foundSubdir, 'should contain subdir1');

% Test 3: Check bytes and isdir for specific entries
for i = 1:n
    if strcmp(listing(i).name, 'file1.txt')
        assert(listing(i).bytes == 5, 'file1.txt should be 5 bytes');
        assert(listing(i).isdir == false, 'file1.txt should not be a dir');
    end
    if strcmp(listing(i).name, 'subdir1')
        assert(listing(i).isdir == true, 'subdir1 should be a dir');
    end
end

% Test 4: Wildcard pattern
listing2 = dir(fullfile(tmpBase, '*.txt'));
assert(length(listing2) == 1, 'should match 1 txt file');
assert(strcmp(listing2(1).name, 'file1.txt'), 'should match file1.txt');

% Test 5: Recursive pattern **
listing3 = dir(fullfile(tmpBase, '**'));
n3 = length(listing3);
foundF1 = false; foundF3 = false; foundF4 = false;
for i = 1:n3
    nm = listing3(i).name;
    if strcmp(nm, 'file1.txt'), foundF1 = true; end
    if strcmp(nm, 'file3.txt'), foundF3 = true; end
    if strcmp(nm, 'file4.dat'), foundF4 = true; end
end
assert(foundF1, '** should find file1.txt');
assert(foundF3, '** should find file3.txt');
assert(foundF4, '** should find file4.dat');

% Test 6: datenum should be a reasonable number (after year 2000)
% datenum for 2000-01-01 is about 730486
for i = 1:n
    if ~strcmp(listing(i).name, '.') && ~strcmp(listing(i).name, '..')
        assert(listing(i).datenum > 730486, 'datenum should be after year 2000');
    end
end

% Test 7: dir with no args lists current directory
listing4 = dir();
assert(length(listing4) >= 2, 'dir() should return at least . and ..');

% Cleanup
rmdir(tmpBase, 's');

disp('SUCCESS');
