% Test zip builtin (round-trips through unzip)

% Setup: a temp tree with a nested directory
srcDir = [tempname(), '_zip_src'];
mkdir(srcDir);
mkdir(fullfile(srcDir, 'sub'));

fid = fopen(fullfile(srcDir, 'a.txt'), 'w');
fprintf(fid, 'alpha');
fclose(fid);

fid = fopen(fullfile(srcDir, 'b.dat'), 'w');
fprintf(fid, 'bravo!');
fclose(fid);

fid = fopen(fullfile(srcDir, 'sub', 'c.txt'), 'w');
fprintf(fid, 'charlie');
fclose(fid);

% zip('base', '.', rootdir) zips the whole tree; .zip is auto-appended
zipBase = [tempname(), '_zip_out'];
entries = zip(zipBase, '.', srcDir);
zipFile = [zipBase '.zip'];
assert(isfile(zipFile), 'zip should create <base>.zip');
assert(length(entries) == 3, 'should zip 3 files');

% Round-trip through unzip and verify contents
outDir = [tempname(), '_zip_extract'];
unzip(zipFile, outDir);
assert(strcmp(fileread(fullfile(outDir, 'a.txt')), 'alpha'));
assert(strcmp(fileread(fullfile(outDir, 'b.dat')), 'bravo!'));
assert(strcmp(fileread(fullfile(outDir, 'sub', 'c.txt')), 'charlie'));

% zip with an explicit file list (cellstr) and a glob
zipFile2 = [tempname(), '_zip_two.zip'];
entries2 = zip(zipFile2, {'a.txt', 'sub'}, srcDir);
assert(length(entries2) == 2, 'a.txt plus recursive sub = 2 files');

zipFile3 = [tempname(), '_zip_glob.zip'];
entries3 = zip(zipFile3, '*.txt', srcDir);
assert(length(entries3) == 1, 'glob *.txt matches only a.txt at top level');

% Cleanup
rmdir(srcDir, 's');
rmdir(outDir, 's');
delete(zipFile);
delete(zipFile2);
delete(zipFile3);

disp('SUCCESS');
