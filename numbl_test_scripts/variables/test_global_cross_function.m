% Test global variables shared across functions
% In MATLAB, global variables are shared across all functions that declare them

global gval;
gval = 0;

% Function modifying global should be visible in caller
inc_global();
assert(gval == 1, 'global modified by function should be 1');

inc_global();
assert(gval == 2, 'global modified again should be 2');

% Function reading global
r = get_global();
assert(r == 2, 'function reading global');

% Set from function, read from script
set_global(42);
assert(gval == 42, 'global set by function');

disp('SUCCESS');

function inc_global()
    global gval;
    gval = gval + 1;
end

function r = get_global()
    global gval;
    r = gval;
end

function set_global(v)
    global gval;
    gval = v;
end
