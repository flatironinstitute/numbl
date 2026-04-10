% Test onCleanup builtin

% Test 1: Basic cleanup runs on normal return
global cleanup_log;
cleanup_log = '';

result = basic_func();
assert(result == 42, 'basic_func return value');
assert(strcmp(cleanup_log, 'cleaned'), 'cleanup should have run');

% Test 2: Cleanup runs on error
cleanup_log = '';
try
    error_func();
catch
end
assert(strcmp(cleanup_log, 'error_cleanup'), 'cleanup should run on error');

% Test 3: Multiple cleanups run in LIFO order
cleanup_log = '';
multi_func();
assert(strcmp(cleanup_log, 'second,first'), 'LIFO order');

% Test 4: Cleanup error does not suppress original return
cleanup_log = '';
result = bad_cleanup_func();
assert(result == 99, 'return value preserved despite cleanup error');

% Test 5: Nested function calls with independent cleanups
cleanup_log = '';
nested_outer();
assert(strcmp(cleanup_log, 'inner,outer'), 'nested cleanups run independently');

disp('SUCCESS');

function result = basic_func()
    global cleanup_log;
    c = onCleanup(@() set_log('cleaned'));
    result = 42;
end

function error_func()
    global cleanup_log;
    c = onCleanup(@() set_log('error_cleanup'));
    error('intentional error');
end

function multi_func()
    global cleanup_log;
    c1 = onCleanup(@() append_log('first'));
    c2 = onCleanup(@() append_log('second'));
end

function result = bad_cleanup_func()
    global cleanup_log;
    c = onCleanup(@() error('cleanup error'));
    result = 99;
end

function nested_outer()
    global cleanup_log;
    c = onCleanup(@() append_log('outer'));
    nested_inner();
end

function nested_inner()
    global cleanup_log;
    c = onCleanup(@() append_log('inner'));
end

function set_log(val)
    global cleanup_log;
    cleanup_log = val;
end

function append_log(val)
    global cleanup_log;
    if isempty(cleanup_log)
        cleanup_log = val;
    else
        cleanup_log = [cleanup_log ',' val];
    end
end
