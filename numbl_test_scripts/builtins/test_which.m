% Test which builtin — resolves a name to its source file path.

% --- Not found returns an empty char array ---
s = which('some_nonexistent_function_xyz_12345');
assert(ischar(s));
assert(isempty(s), sprintf('expected empty, got "%s"', s));

% --- Variable in current workspace ---
myvar = 42;
s = which('myvar');
assert(ischar(s));
assert(strcmp(s, 'variable'), ...
    sprintf('expected "variable", got "%s"', s));

% --- Builtin function — MATLAB returns "built-in (<path>)", numbl returns
%     just "built-in".  Accept either.
s = which('sin');
assert(ischar(s));
assert(~isempty(s), 'which(sin) should not be empty');
assert(contains(s, 'built-in'), ...
    sprintf('expected built-in report, got "%s"', s));

% --- Workspace function on the path ---
[thisDir, ~, ~] = fileparts(mfilename('fullpath'));
libDir = fullfile(thisDir, 'lib_which_test');
addpath(libDir);
try
    s = which('which_test_helper');
    assert(ischar(s));
    assert(~isempty(s), 'which_test_helper should be found on path');
    % In both MATLAB and numbl the returned string is the .m file path.
    assert(contains(s, 'which_test_helper') && endsWith(s, '.m'), ...
        sprintf('expected .m file path, got "%s"', s));
catch ME
    rmpath(libDir);
    rethrow(ME);
end
rmpath(libDir);

% After rmpath it should be empty again.
s = which('which_test_helper');
assert(isempty(s), 'which_test_helper should be empty after rmpath');

disp('SUCCESS');
