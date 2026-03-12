% Test import precedence rules

% Explicit import beats local functions
import mymath.add_two

% This local function should be shadowed by the explicit import
result = add_two(3, 5);
assert(result == 8, 'Explicit import should win over local function');

% Wildcard import loses to local functions
import utils.*
result2 = double_it(7);
assert(result2 == 700, 'Local function should win over wildcard import');

disp('SUCCESS')

function result = double_it(x)
    result = x * 100;
end
