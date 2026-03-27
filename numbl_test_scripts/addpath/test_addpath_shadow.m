% Test that addpath to front shadows existing paths

[thisDir, ~, ~] = fileparts(mfilename('fullpath'));
dirA = fullfile(thisDir, 'lib_a');
dirShadow = fullfile(thisDir, 'lib_shadow');

% Add lib_a to end, then lib_shadow to front — shadow should win
addpath(dirA, '-end');
addpath(dirShadow);
assert(shared_func(5) == 495, 'lib_shadow shared_func should win (x*99)');

% Remove shadow — lib_a should take over
rmpath(dirShadow);
assert(shared_func(5) == 50, 'lib_a shared_func should take over (x*10)');

% Clean up
rmpath(dirA);

disp('SUCCESS');
