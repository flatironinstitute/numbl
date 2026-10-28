% arguments blocks: Repeating arguments and (Output) validation.
% Verified against MATLAB R2025b.

% --- Repeating: each repeating argument collects a 1-by-N cell ---
c = rep(1, 2, 3, 4);
assert(isequal(c{1}, {1, 3}) && isequal(c{2}, {2, 4}));
c = rep();
assert(isequal(size(c{1}), [1 0]) && iscell(c{1}));
c = rep(1, 2, 'a', 4);
assert(isequal(c{1}, {1, 97}));
check_err(@() rep(1, 2, 3), 'MATLAB:functionValidation:RepeatingInputsNotInGroups', ...
  'Invalid argument list. Repeating arguments, if specified, must be in groups of 2.');
check_err(@() rep(1, 2, 3, -4), 'MATLAB:validators:mustBePositive', ...
  'Invalid argument at position 4. Value must be positive.');
assert(isequal(rep2(5, 1, 2), {1, 2}));
check_err(@() rep2(5, 1, -2), 'MATLAB:validators:mustBePositive', ...
  'Invalid argument at position 3. Value must be positive.');
assert(isequal(repva(1, 2, 3), {2, 3}));

% --- Repeating followed by name-value arguments ---
c = repnv(1, 2, 'scale', 4);
assert(isequal(c{1}, {1, 2}) && c{2} == 4);
c = repnv('scale', 4);
assert(isempty(c{1}) && c{2} == 4);
c = repnv(1, 'x', 4);
assert(isequal(c{1}, {1, 'x', 4}) && c{2} == 1);

% --- Output validation ---
check_err(@() outv(-1), 'MATLAB:validators:mustBePositive', ...
  'Invalid output ''y''. Value must be positive.');
y = outv(true);
assert(isa(y, 'double') && y == 1);
check_err(@() outsz([1 2]), 'MATLAB:validation:IncompatibleSize', ...
  'Invalid output ''y''. Value must be a scalar.');
% every assigned output is validated, even when not requested
check_err(@() outv2(), 'MATLAB:validators:mustBeNumeric', ...
  'Invalid output ''b''. Value must be numeric.');
[a, b] = outv3();
assert(a == 1 && isequal(b, [1 2 3]));

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

function o = rep(x, y)
  arguments (Repeating)
    x double
    y double {mustBePositive}
  end
  o = {x, y};
end

function o = rep2(a, v)
  arguments
    a
  end
  arguments (Repeating)
    v {mustBePositive}
  end
  o = v;
  o{1} = o{1} + 0 * a;
end

function o = repva(a, varargin)
  arguments
    a
  end
  arguments (Repeating)
    varargin
  end
  o = varargin;
  o{1} = o{1} + 0 * a;
end

function o = repnv(x, opts)
  arguments (Repeating)
    x
  end
  arguments
    opts.scale = 1
  end
  o = {x, opts.scale};
end

function y = outv(x)
  arguments
    x
  end
  arguments (Output)
    y (1,1) double {mustBePositive}
  end
  y = x;
end

function y = outsz(x)
  arguments
    x
  end
  arguments (Output)
    y (1,1)
  end
  y = x;
end

function [a, b] = outv2()
  arguments (Output)
    a
    b {mustBeNumeric}
  end
  a = 1;
  b = 'x';
end

function [a, b] = outv3()
  arguments (Output)
    a (1,1) double
    b (1,:) double
  end
  a = true;
  b = [1; 2; 3];
end
