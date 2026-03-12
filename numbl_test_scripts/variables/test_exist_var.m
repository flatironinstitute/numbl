% Test exist() with 'var' option

% Variable that exists
x = 5;
assert(exist('x', 'var') == 1, 'exist should return 1 for defined variable');

% Variable that does not exist
assert(exist('nonexistent_var_xyz', 'var') == 0, 'exist should return 0 for undefined variable');

% After assignment in a loop
for i = 1:3
end
assert(exist('i', 'var') == 1, 'exist for loop variable');

% Cell array variable
c = {1, 2};
assert(exist('c', 'var') == 1, 'exist for cell variable');

% Struct variable
s.a = 1;
assert(exist('s', 'var') == 1, 'exist for struct variable');

disp('SUCCESS');
