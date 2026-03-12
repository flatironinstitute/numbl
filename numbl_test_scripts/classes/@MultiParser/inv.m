function result = inv(obj, varargin)
    % A separate method file with its own local parseInputs (3 outputs).
    % This mirrors @chebfun/inv.m having its own parseInputs.
    % Crucially, this method also creates a new MultiParser instance,
    % which triggers constructor lowering while inv's parseInputs is active.
    [tol, opts, pref] = parseInputs(obj, varargin{:});
    result = MultiParser(tol, opts);
end

function [tol, opts, pref] = parseInputs(f, varargin)
    tol = f.val1;
    opts = 2;
    pref = 3;
end
