% Test adding multiple directories

[thisDir, ~, ~] = fileparts(mfilename('fullpath'));
dirA = fullfile(thisDir, 'lib_a');
dirB = fullfile(thisDir, 'lib_b');

% Add both directories
addpath(dirA);
addpath(dirB);

% Both functions should be available
assert(helper_a(4) == 8, 'helper_a should work');
assert(helper_b(4) == 104, 'helper_b should work');

% Remove one — the other should still work
rmpath(dirA);

try
    helper_a(4);
    error('helper_a should be gone');
catch
end

assert(helper_b(4) == 104, 'helper_b should still work after rmpath(dirA)');

% Clean up
rmpath(dirB);

disp('SUCCESS');
