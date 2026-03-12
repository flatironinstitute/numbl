% Test exist() with a single argument

% Variable in workspace -> 1
myvar = 42;
assert(exist('myvar') == 1, 'defined variable should return 1');

% Variable that does not exist -> 0
assert(exist('undefined_var_xyz') == 0, 'undefined variable should return 0');

% Built-in function -> 5 (when no variable shadows it)
assert(exist('sin') == 5, 'sin should return 5 (builtin)');
assert(exist('cos') == 5, 'cos should return 5 (builtin)');
assert(exist('zeros') == 5, 'zeros should return 5 (builtin)');

% Variable shadows builtin: if a var named 'zeros' is defined, return 1
zeros_var = 99;
assert(exist('zeros_var') == 1, 'defined variable should return 1');

% Completely unknown name -> 0
assert(exist('totally_nonexistent_abc123') == 0, 'unknown name should return 0');

disp('SUCCESS');
