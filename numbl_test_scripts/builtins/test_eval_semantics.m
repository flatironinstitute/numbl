% eval runs in the caller's workspace and shares the run's state.
x = 1;
eval('x = x + 1;');
assert(x == 2);

eval('1 + 2;');
assert(ans == 3);

y = eval('x * 10');
assert(y == 20);

[r, c] = eval('size(ones(2, 3))');
assert(r == 2 && c == 3);

% Two-argument form runs the catch text on error.
eval('error(''boom'');', 'caught = true;');
assert(caught);

% Errors propagate to try/catch.
try
    eval('undefined_function_xyz(1);');
    assert(false);
catch err
    assert(~isempty(err.message));
end

% Global variables and warning state are shared with the caller.
global G
G = 5;
eval('global G; G = G + 1;');
assert(G == 6);
eval('warning(''off'', ''my:id'');');
s = warning('query', 'my:id');
assert(strcmp(s.state, 'off'));
warning('on', 'my:id');

% Inside a function, eval sees the function's locals.
assert(helper(4) == 40);
disp('SUCCESS')

function r = helper(a)
    eval('r = a * 10;');
end
