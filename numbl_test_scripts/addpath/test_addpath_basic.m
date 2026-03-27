% Test basic addpath and function resolution

[thisDir, ~, ~] = fileparts(mfilename('fullpath'));
libDir = fullfile(thisDir, 'lib_a');

% helper_a should not be available before addpath
try
    helper_a(5);
    error('Should not reach here');
catch
    % Expected
end

% Add path and call function
addpath(libDir);
result = helper_a(5);
assert(result == 10, 'helper_a(5) should return 10');

% Clean up
rmpath(libDir);

disp('SUCCESS');
