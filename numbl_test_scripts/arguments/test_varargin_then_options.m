% Test varargin followed by a name-value options parameter
% (arguments blocks: Repeating varargin + options struct)

[a, extras, s] = f(1, 2, 3);
assert(a == 1);
assert(isequal(extras, {2, 3}));
assert(s == 1);

[a, extras, s] = f(1, 2, 'scale', 10);
assert(a == 1);
assert(isequal(extras, {2}));
assert(s == 10);

[a, extras, s] = f(1, 'scale', 10);
assert(a == 1);
assert(isempty(extras));
assert(s == 10);

% Name=Value call syntax routes to the options struct too
[a, extras, s] = f(1, 2, scale=7);
assert(isequal(extras, {2}));
assert(s == 7);

disp('SUCCESS')

function [a, extras, s] = f(a, varargin, opts)
    arguments
        a
    end
    arguments (Repeating)
        varargin
    end
    arguments
        opts.scale = 1
    end
    extras = varargin;
    s = opts.scale;
end
