% Test that caught MException struct exposes a stack field matching
% MATLAB's shape: a struct array with file, name, line fields, ordered
% innermost-first.  This is relied on by +mip/test.m to report where a
% test script actually failed rather than just the mip wrapper site.

function inner_fn()
    error('my:id', 'boom from inner');
end

function outer_fn()
    inner_fn();
end

caught = false;
try
    outer_fn();
catch ME
    caught = true;

    % Basic fields.
    assert(ischar(ME.message), 'message should be char');
    assert(strcmp(ME.message, 'boom from inner'), 'message value');
    assert(strcmp(ME.identifier, 'my:id'), 'identifier value');

    % Stack field.  Note: in MATLAB, ME is an MException object, so
    % isfield(ME,...) returns false even though ME.stack exists.  We
    % access ME.stack directly instead.
    assert(~isempty(ME.stack), 'stack should be non-empty');
    assert(numel(ME.stack) >= 2, 'stack should have at least 2 frames');

    % Innermost frame is inner_fn.
    assert(strcmp(ME.stack(1).name, 'inner_fn'), 'frame 1 name');
    assert(ME.stack(1).line > 0, 'frame 1 line > 0');
    assert(~isempty(ME.stack(1).file), 'frame 1 file non-empty');

    % Next frame is outer_fn.
    assert(strcmp(ME.stack(2).name, 'outer_fn'), 'frame 2 name');
    assert(ME.stack(2).line > 0, 'frame 2 line > 0');

    % Iterate — should not error regardless of number of extra frames
    % (MATLAB may append a `run` frame, numbl may append an empty-name
    % top-level frame; either is tolerated).  ME.stack itself is a
    % struct array in both, so isfield DOES work on it.
    assert(isfield(ME.stack, 'file'), 'stack has file');
    assert(isfield(ME.stack, 'name'), 'stack has name');
    assert(isfield(ME.stack, 'line'), 'stack has line');
    for k = 1:numel(ME.stack)
        frame = ME.stack(k);
        assert(ischar(frame.file), 'file is char');
        assert(ischar(frame.name), 'name is char');
        assert(isnumeric(frame.line), 'line is numeric');
    end
end
assert(caught, 'catch should have triggered');

% Error thrown from top-level (no function frames) still has a usable
% stack field.
caught2 = false;
try
    error('top:id', 'top-level boom');
catch ME2
    caught2 = true;
    assert(strcmp(ME2.identifier, 'top:id'), 'top identifier');
    % stack may be empty or contain a single frame pointing at this file;
    % either is acceptable, but accessing it must not throw.
    n = numel(ME2.stack);
    assert(n >= 0, 'stack numel valid');
end
assert(caught2, 'top-level catch should have triggered');

disp('SUCCESS');
