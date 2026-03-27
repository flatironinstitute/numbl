% Test addpath with '-end' flag (append to end of path)

[thisDir, ~, ~] = fileparts(mfilename('fullpath'));
dirA = fullfile(thisDir, 'lib_a');
dirShadow = fullfile(thisDir, 'lib_shadow');

% Add lib_a first (has shared_func returning x*10)
addpath(dirA);
assert(shared_func(5) == 50, 'lib_a shared_func should return 50');

% Add lib_shadow to end — lib_a should still win (first-wins priority)
addpath(dirShadow, '-end');
assert(shared_func(5) == 50, 'lib_a should still win with -end');

% Clean up
rmpath(dirA);
rmpath(dirShadow);

disp('SUCCESS');
