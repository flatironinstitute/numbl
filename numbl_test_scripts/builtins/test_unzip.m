% Test unzip builtin

% Get the directory of this test script
[thisDir, ~, ~] = fileparts(mfilename('fullpath'));
zipFile = fullfile(thisDir, 'test_unzip_fixture.zip');
outDir = 'tmp_test_unzip_output';

% Clean up from any previous run
if exist(outDir, 'dir')
    rmdir(outDir, 's');
end

% Test basic extraction with output capture
filenames = unzip(zipFile, outDir);
assert(iscell(filenames), 'unzip should return a cell array');
assert(length(filenames) == 2, 'should extract 2 files');

% Verify extracted file contents
txt1 = fileread(fullfile(outDir, 'hello.txt'));
assert(strcmp(txt1, 'Hello World'), 'hello.txt content mismatch');

txt2 = fileread(fullfile(outDir, 'subdir', 'nested.txt'));
assert(strcmp(txt2, 'Nested content'), 'nested.txt content mismatch');

% Clean up
rmdir(outDir, 's');

% Test extraction without output capture
unzip(zipFile, 'tmp_test_unzip_output2');
txt = fileread(fullfile('tmp_test_unzip_output2', 'hello.txt'));
assert(strcmp(txt, 'Hello World'), 'extraction without nargout failed');
rmdir('tmp_test_unzip_output2', 's');

disp('SUCCESS');
