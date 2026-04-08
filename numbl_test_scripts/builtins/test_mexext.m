% Test mexext builtin — returns the mex-file extension for the current
% platform.

e = mexext;
assert(ischar(e));
assert(~isempty(e));

% Should start with 'mex' and be a known extension.
valid = {'mexa64', 'mexmaci64', 'mexmaca64', 'mexw64'};
assert(any(strcmp(e, valid)), sprintf('unexpected mexext: %s', e));

% Platform-specific sanity check: on Linux we expect mexa64.
if isunix && ~ismac
    assert(strcmp(e, 'mexa64'), sprintf('expected mexa64, got %s', e));
end

% Windows expects mexw64.
if ispc
    assert(strcmp(e, 'mexw64'), sprintf('expected mexw64, got %s', e));
end

% macOS expects mexmaca64 (Apple Silicon) or mexmaci64 (Intel).
if ismac
    assert(strcmp(e, 'mexmaca64') || strcmp(e, 'mexmaci64'), ...
        sprintf('expected mac mex ext, got %s', e));
end

disp('SUCCESS');
