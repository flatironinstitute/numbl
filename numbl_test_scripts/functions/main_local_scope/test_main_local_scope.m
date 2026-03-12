% Test: main script's local functions should NOT leak into workspace functions
% In MATLAB, local functions are scoped to their file. A workspace function
% calling a name that matches both a main-script local function and a
% workspace function should use the workspace function.

% From main script, local function should shadow workspace function
r1 = shared_name(5);
assert(r1 == -1, 'main script should use its own local function');

% From workspace function, workspace function should be used (not main local)
r2 = ws_caller(5);
assert(r2 == 500, 'workspace function should use workspace shared_name, not main local');

disp('SUCCESS');

function r = shared_name(~)
    r = -1;
end
