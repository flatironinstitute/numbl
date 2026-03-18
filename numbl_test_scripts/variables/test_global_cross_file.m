% Test global variables across separately-defined function files

% Test 1: Basic cross-function global communication
global shared_var;
shared_var = 0;
set_shared(42);
v = get_shared();
assert(v == 42);

% Test 2: Multiple globals
global gA gB;
gA = 10;
gB = 20;
swap_globals();
assert(gA == 20);
assert(gB == 10);

disp('SUCCESS');

function set_shared(val)
    global shared_var;
    shared_var = val;
end

function v = get_shared()
    global shared_var;
    v = shared_var;
end

function swap_globals()
    global gA gB;
    tmp = gA;
    gA = gB;
    gB = tmp;
end
