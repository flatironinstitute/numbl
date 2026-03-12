% Test that nargoutOverride does not leak into on-demand compiled functions
% Bug: when [varargout{:}] = foo(...) compiles a function on-demand,
% the nargout temp variable from the varargout context leaks into the
% newly compiled function, causing "$tXXX is not defined" errors.

function test_nargout_override_leak()
    % Call wrapper which uses varargout, triggering on-demand compilation
    % of helper_func with a nargoutOverride active
    result = wrapper('hello');
    assert(result == 1);
    disp('SUCCESS');
end

function varargout = wrapper(x)
    % This varargout pattern sets nargoutOverride to a temp var
    % When helper_func is compiled on-demand during this call,
    % the nargoutOverride must not leak into helper_func's codegen
    [varargout{1:nargout}] = helper_func(x);
end

function out = helper_func(x)
    % This function uses numel() internally.
    % If nargoutOverride leaks, numel will get $tXXX instead of 1 as nargout
    out = numel(x) > 0;
end
