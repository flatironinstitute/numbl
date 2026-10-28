% arguments blocks: name-value arguments (opts.Name) with defaults and
% validation, name matching, and argument-list errors. Verified against
% MATLAB R2025b.

o = nv(1);
assert(o.alpha == 1 && o.beta == 2);
o = nv(1, 'alpha', 5);
assert(o.alpha == 5);
% names match case-insensitively, and by unique prefix
o = nv(1, 'Alpha', 5);
assert(o.alpha == 5);
o = nv(1, 'alp', 6);
assert(o.alpha == 6);
o = nv(1, 'a', 7);
assert(o.alpha == 7);
o = nv(1, "beta", 8);
assert(o.beta == 8);
o = nv(1, alpha=9);
assert(o.alpha == 9);
% the last occurrence wins
o = nv(1, 'alpha', 2, 'alpha', 3);
assert(o.alpha == 3);
% class conversion
o = nv(1, 'alpha', 'x');
assert(o.alpha == 120);
% fields are in declaration order
o = nv(1, 'beta', 1, 'alpha', 2);
assert(isequal(fieldnames(o), {'alpha'; 'beta'}));

check_err(@() nv(1, 'gamma', 5), 'MATLAB:TooManyInputs', ...
  'Invalid argument name ''gamma''. Name must be ''alpha'' or ''beta''.');
check_err(@() nv3(1, 'd', 5), 'MATLAB:TooManyInputs', ...
  'Invalid argument name ''d''. Name must be ''a1'', ''b1'', or ''c1''.');
check_err(@() nvtwo(1, 'a', 3), 'MATLAB:functionValidation:AmbiguousInputName', ...
  'Ambiguous argument name ''a''. Name must be ''alpha'' or ''abc''.');
check_err(@() nv(1, 'alpha'), 'MATLAB:TooManyInputs', ...
  'Invalid argument at position 2. Make sure name-value arguments include both the name and the value.');
check_err(@() nv(1, 5, 5), 'MATLAB:TooManyInputs', ...
  'Invalid argument at position 2. Function requires exactly 1 positional input(s).');
check_err(@() nv(1, struct('alpha', 4)), 'MATLAB:TooManyInputs', ...
  'Invalid argument at position 2. Function requires exactly 1 positional input(s).');
check_err(@() nv(1, 'alpha', -5), 'MATLAB:validators:mustBePositive', ...
  'Invalid value for ''alpha'' argument. Value must be positive.');
check_err(@() nv(1, 'alpha', [1 2]), 'MATLAB:validation:IncompatibleSize', ...
  'Invalid value for ''alpha'' argument. Value must be a scalar.');
check_err(@() nvdefbad(1), 'MATLAB:validators:mustBePositive', ...
  'Invalid default value for argument ''alpha''. Value must be positive.');

% a name-value argument without a default is absent unless passed
o = nvreq(1);
assert(~isfield(o, 'alpha') && o.beta == 2);
o = nvreq(1, 'alpha', 4);
assert(o.alpha == 4);

% defaults and validators may reference earlier arguments
o = nvdefref(3);
assert(o.scale == 6);
o = nvval(1, 'lim', 20);
assert(o.lim == 20);
check_err(@() nvval(1, 'lim', 0), 'MATLAB:validators:mustBeGreaterThan', ...
  'Invalid value for ''lim'' argument. Value must be greater than 1.');

% an optional positional argument before the name-value arguments
assert(isequal(nvopt(1), {1, 10, 1}));
assert(isequal(nvopt(1, 2), {1, 2, 1}));
assert(isequal(nvopt(1, 'scale', 3), {1, 10, 3}));
assert(isequal(nvopt(1, 'sc', 4), {1, 10, 4}));
assert(isequal(nvopt(1, 2, 'scale', 3), {1, 2, 3}));
assert(isequal(nvopt(1, 'foo'), {1, 'foo', 1}));
assert(isequal(nvopt(1, 'scale'), {1, 'scale', 1}));
check_err(@() nvopt(1, 2, 3), 'MATLAB:TooManyInputs', ...
  'Invalid argument at position 3. Function requires 1 to 2 positional input(s).');

% name-value arguments of a class method and a static method
r = NvOptsVal_.scaled(2, 'factor', 3);
assert(r == 6);
check_err(@() NvOptsVal_.scaled(2, 'factor', -3), 'MATLAB:validators:mustBePositive', ...
  'Invalid value for ''factor'' argument. Value must be positive.');

disp('SUCCESS')

function check_err(f, id, msg)
  try
    f();
  catch e
    assert(strcmp(e.identifier, id), sprintf('identifier: got [%s], expected [%s]', e.identifier, id));
    assert(strcmp(e.message, msg), sprintf('message: got [%s], expected [%s]', e.message, msg));
    return;
  end
  error('expected an error: %s', msg);
end

function o = nv(x, opts)
  arguments
    x
    opts.alpha (1,1) double {mustBePositive} = 1
    opts.beta = 2
  end
  o = opts;
  o.beta = o.beta + 0 * x;
end

function o = nv3(x, opts)
  arguments
    x
    opts.a1 = 1
    opts.b1 = 2
    opts.c1 = 3
  end
  o = opts;
  o.a1 = o.a1 + 0 * x;
end

function o = nvtwo(x, opts)
  arguments
    x
    opts.alpha = 1
    opts.abc = 2
  end
  o = opts;
  o.abc = o.abc + 0 * x;
end

function o = nvdefbad(x, opts)
  arguments
    x
    opts.alpha {mustBePositive} = -1
  end
  o = opts;
  o.alpha = o.alpha + 0 * x;
end

function o = nvreq(x, opts)
  arguments
    x
    opts.alpha
    opts.beta = 2
  end
  o = opts;
  o.beta = o.beta + 0 * x;
end

function o = nvdefref(x, opts)
  arguments
    x
    opts.scale = x * 2
  end
  o = opts;
end

function o = nvval(x, opts)
  arguments
    x
    opts.lim {mustBeGreaterThan(opts.lim, x)} = 5
  end
  o = opts;
end

function o = nvopt(x, y, opts)
  arguments
    x
    y = 10
    opts.scale = 1
  end
  o = {x, y, opts.scale};
end
