%% inputParser: basic optional arguments with defaults
p = inputParser;
addOptional(p, 'x', 10);
addOptional(p, 'y', 20);
parse(p);
assert(p.Results.x == 10, 'default x');
assert(p.Results.y == 20, 'default y');

%% inputParser: parse with values
p2 = inputParser;
addOptional(p2, 'x', 10);
addOptional(p2, 'y', 20);
parse(p2, 5, 15);
assert(p2.Results.x == 5, 'parsed x');
assert(p2.Results.y == 15, 'parsed y');

%% inputParser: required arguments
p3 = inputParser;
addRequired(p3, 'name');
addOptional(p3, 'age', 0);
parse(p3, 'hello', 42);
assert(strcmp(p3.Results.name, 'hello'), 'required name');
assert(p3.Results.age == 42, 'optional age');

%% inputParser: name-value pairs (addParameter)
p4 = inputParser;
addOptional(p4, 'x', 1);
addParameter(p4, 'color', 'red');
parse(p4, 5, 'color', 'blue');
assert(p4.Results.x == 5, 'optional x with params');
assert(strcmp(p4.Results.color, 'blue'), 'param color');

%% inputParser: defaults for name-value pairs
p5 = inputParser;
addParameter(p5, 'alpha', 0.5);
addParameter(p5, 'beta', 1.0);
parse(p5);
assert(p5.Results.alpha == 0.5, 'default alpha');
assert(p5.Results.beta == 1.0, 'default beta');

%% inputParser: skip optional when name-value follows
p6 = inputParser;
addOptional(p6, 'x', 1);
addParameter(p6, 'mode', 'fast');
parse(p6, 'mode', 'slow');
assert(p6.Results.x == 1, 'optional x defaulted');
assert(strcmp(p6.Results.mode, 'slow'), 'param mode');

%% inputParser: KeepUnmatched
p7 = inputParser;
p7.KeepUnmatched = true;
addParameter(p7, 'a', 1);
parse(p7, 'a', 10, 'b', 20);
assert(p7.Results.a == 10, 'matched param');
assert(p7.Unmatched.b == 20, 'unmatched param');

%% struct array field assignment (related fix)
s = struct('name', 'hello', 'validator', {{}});
s.validator = @isnumeric;
assert(strcmp(s.name, 'hello'), 'struct array field preserved after assignment');
assert(isa(s.validator, 'function_handle'), 'assigned validator');

disp('SUCCESS');
