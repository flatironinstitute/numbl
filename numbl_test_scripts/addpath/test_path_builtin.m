% Test the path() builtin

[thisDir, ~, ~] = fileparts(mfilename('fullpath'));
dirA = fullfile(thisDir, 'lib_a');

% Get initial path
p0 = path();

% Add a directory
addpath(dirA);

% path() should now contain lib_a
p1 = path();
assert(~isempty(strfind(p1, 'lib_a')), 'path should contain lib_a after addpath');

% Remove it
rmpath(dirA);

% path() should no longer contain lib_a
p2 = path();
assert(isempty(strfind(p2, 'lib_a')), 'path should not contain lib_a after rmpath');

disp('SUCCESS');
