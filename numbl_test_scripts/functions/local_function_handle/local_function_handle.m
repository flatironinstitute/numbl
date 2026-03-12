% Test returning a handle to a local function from a workspace function
% and calling it from the main script.

% Get a handle that wraps a local function
h = get_local_handle(3);

% Call the handle from the main script
result = h(5);
assert(result == 15, 'Expected 15');

% Test with a different scale
h2 = get_local_handle(10);
assert(h2(4) == 40, 'Expected 40');

% Test with array input
h3 = get_local_handle(2);
assert(isequal(h3([1 2 3]), [2 4 6]), 'Expected [2 4 6]');

disp('SUCCESS');
