% Behavior of functions declared with no return arguments.
%
% In MATLAB, a function declared with no outputs:
%   - does not set `ans` when called without a semicolon
%   - does not print anything beyond what the function body prints
%   - errors with "Too many output arguments." if the call site expects
%     any outputs (e.g. `r = fn()` or nested inside another call)

function no_output_fn(x)
    assert(x == 5, 'argument passed through');
end

% Seed ans with a known sentinel value.
42;

% Call with a semicolon: ans must not be overwritten.
no_output_fn(5);
assert(ans == 42, 'semicolon call should not overwrite ans');

% Call without a semicolon: ans must also not be overwritten.
no_output_fn(5)
assert(ans == 42, 'no-semicolon call should not overwrite ans');

% Assigning the result of a no-output function errors.
err = '';
try
    r = no_output_fn(5); %#ok<NASGU>
catch ME
    err = ME.message;
end
assert(strcmp(err, 'Too many output arguments.'), ...
    'assigning no-output function should raise "Too many output arguments."');

% Using a no-output function inside another call errors the same way.
err = '';
try
    disp(no_output_fn(5))
catch ME
    err = ME.message;
end
assert(strcmp(err, 'Too many output arguments.'), ...
    'nesting no-output function should raise "Too many output arguments."');

% feval of a no-output function works (nargout=0 at the call site).
42;
feval(@no_output_fn, 5);
assert(ans == 42, 'feval should not overwrite ans either');

disp('SUCCESS');
