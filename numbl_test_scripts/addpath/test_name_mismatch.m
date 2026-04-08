% Test that a .m function file whose declared function name differs from
% its file name is still callable by the file name (MATLAB behavior).
%
% MATLAB issues a warning but runs the function body — numbl used to
% silently fall through to a script-mode path that ignored args and the
% body entirely, causing functions like chunkie's chnk.intchunk.fcoefs
% (file name fcoefs.m, declared chunkerintchunk_fcoefs) to no-op.

[thisDir, ~, ~] = fileparts(mfilename('fullpath'));
libDir = fullfile(thisDir, 'lib_name_mismatch');

% Silence the "function name mismatch" warning MATLAB emits on addpath.
warning('off', 'MATLAB:dispatcher:nameConflict');

addpath(libDir);
try
    % File: mismatched_file.m, declared: function y = some_other_name(a,b)
    % Calling by file name should execute the body and return 10*a + b.
    y = mismatched_file(3, 7);
    assert(y == 37, sprintf('expected 37, got %g', y));

    % Varying args
    y2 = mismatched_file(1, 2);
    assert(y2 == 12, sprintf('expected 12, got %g', y2));

    % File: mismatch_no_output.m, declared: function declared_differently(x)
    % No return value; body asserts x == 7.  Should not throw.
    mismatch_no_output(7);

    % Should throw if we pass a value that fails the inner assert
    threw = false;
    try
        mismatch_no_output(99);
    catch
        threw = true;
    end
    assert(threw, 'inner assert should have fired');
catch ME
    rmpath(libDir);
    warning('on', 'MATLAB:dispatcher:nameConflict');
    rethrow(ME);
end
rmpath(libDir);
warning('on', 'MATLAB:dispatcher:nameConflict');

disp('SUCCESS');
