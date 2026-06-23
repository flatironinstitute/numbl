% Test that declaring a global that has never been assigned makes it
% readable as [] (MATLAB semantics), rather than erroring as undefined.

%% Read an unassigned global at script level
global gnever;
assert(isempty(gnever), 'unassigned global should be [] (empty)');

%% Unassigned global initialized lazily on first read inside a function
global gunset;
read_then_set();
assert(gunset == 42, 'global set inside function after empty read should be readable');

%% Pattern from hodlroption: default-fill an unset global
get_option();
global opt_threshold;
assert(opt_threshold == 1e-12, 'default should be filled when global was empty');

disp('SUCCESS');

function read_then_set()
    global gunset;
    assert(isempty(gunset), 'unassigned global should be empty inside function');
    gunset = 42;
end

function get_option()
    global opt_threshold;
    if isempty(opt_threshold)
        opt_threshold = 1e-12;
    end
end
