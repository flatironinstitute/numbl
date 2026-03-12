% Test advanced global variable scenarios

%% Multiple global variables
global ga gb;
ga = 10;
gb = 20;
swap_globals();
assert(ga == 20, 'ga should be swapped to 20');
assert(gb == 10, 'gb should be swapped to 10');

%% Global with non-scalar values (arrays)
global garr;
garr = [1, 2, 3];
append_to_global();
assert(isequal(garr, [1, 2, 3, 4]), 'global array should have 4 appended');

%% Global initialized inside function, read from script
global gfunc_init;
init_global_val();
assert(gfunc_init == 99, 'global initialized in function should be readable from script');

%% Two functions communicating through global without script involvement
global gcomm;
gcomm = 0;
sender();
v = receiver();
assert(v == 777, 'receiver should read value set by sender');

%% Global with struct value
global gstruct;
gstruct = struct('x', 1, 'y', 2);
modify_global_struct();
assert(gstruct.x == 10, 'global struct field x should be modified');
assert(gstruct.y == 2, 'global struct field y should be unchanged');

%% Global with string value
global gstr;
gstr = 'hello';
append_global_str();
assert(strcmp(gstr, 'hello world'), 'global string should be appended');

disp('SUCCESS');

function swap_globals()
    global ga gb;
    tmp = ga;
    ga = gb;
    gb = tmp;
end

function append_to_global()
    global garr;
    garr = [garr, 4];
end

function init_global_val()
    global gfunc_init;
    gfunc_init = 99;
end

function sender()
    global gcomm;
    gcomm = 777;
end

function v = receiver()
    global gcomm;
    v = gcomm;
end

function modify_global_struct()
    global gstruct;
    gstruct.x = 10;
end

function append_global_str()
    global gstr;
    gstr = [gstr, ' world'];
end
