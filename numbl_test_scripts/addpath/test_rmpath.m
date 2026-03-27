% Test rmpath removes function access

[thisDir, ~, ~] = fileparts(mfilename('fullpath'));
libDir = fullfile(thisDir, 'lib_a');

% Add, verify, remove, verify gone
addpath(libDir);
assert(helper_a(3) == 6, 'helper_a should work after addpath');

rmpath(libDir);

try
    helper_a(3);
    error('Should not reach here after rmpath');
catch
    % Expected: function no longer available
end

disp('SUCCESS');
