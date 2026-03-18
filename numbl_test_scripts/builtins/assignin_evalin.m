% Test assignin and evalin with workspace

x = 10;

% Test evalin reads workspace variable
assert(test_evalin_read() == 10);

% Test assignin sets workspace variable
test_assignin_write();
assert(x == 42);

% Test evalin with default value for nonexistent variable
assert(test_evalin_default() == 99);

% Test evalin error for nonexistent variable without default
try
    test_evalin_error();
    assert(false, 'Should have thrown an error');
catch e
    assert(contains(e.message, 'does not exist'));
end

% Test assignin/evalin with caller scope
y = 100;
test_caller_outer(y);
assert(y == 100); % y in workspace should not be affected

% Test caller modifies the caller's variable
result = test_caller_modify();
assert(result == 999);

% Test caller from main script: function reads main script variable via caller
w = 77;
assert(caller_reader_w() == 77);

% Test caller from main script: function writes main script variable via caller
caller_writer_w();
assert(w == 888);

% ── Type inference edge cases ──────────────────────────────────────────

% Edge case 1: workspace variable type changes from scalar to vector
% If type isn't reset, native math (x + 1) would assume scalar and fail
x = 5;
ws_set_x_vector();  % sets x to [2, 5] via assignin('workspace', ...)
result = x + 1;
assert(isequal(result, [3, 6]));

% Edge case 2: workspace variable type changes from number to string
x = 10;
ws_set_x_string();  % sets x to 'hello' via assignin('workspace', ...)
assert(isequal(x, 'hello'));

% Edge case 3: workspace evalin after type change — make sure we can read
% the new type correctly
x = 42;
ws_set_x_vector();  % sets x to [2, 5]
val = test_evalin_read();  % reads x from workspace
assert(isequal(val, [2, 5]));

% Edge case 4: multiple workspace variables, only one changes type
x = 1;
a = 2;
ws_set_x_vector();  % only x changes
result_a = a + 10;  % a should still be scalar
result_x = x + 1;   % x is now a vector
assert(result_a == 12);
assert(isequal(result_x, [3, 6]));

% Edge case 5: caller type change — scalar to matrix
result2 = test_caller_type_change();
assert(isequal(result2, [2, 4; 6, 8]));

% Edge case 6: caller type change — number to struct
result3 = test_caller_struct_change();
assert(result3.value == 42);

% Edge case 7: chained calls — caller sets variable, then another
% function reads it back via caller with the new type
result4 = test_caller_chain();
assert(isequal(result4, [10, 20, 30]));

% Edge case 8: assignin('workspace', ...) inside a loop
x = 0;
for i = 1:3
    ws_set_x_to(i * 10);  % sets x via assignin('workspace', ...)
end
assert(x == 30);

% Edge case 9: evalin('caller', ...) with default when variable exists
% should return the actual value, not the default
result5 = test_caller_default_exists();
assert(result5 == 7);

% Edge case 10: workspace variable read after conditional assignin
x = 100;
ws_set_x_conditional(true);
assert(x == -1);
ws_set_x_conditional(false);
assert(x == -1);  % unchanged since condition was false

% Edge case 11: variable created by assignin('caller') in a directly-called function
caller_create_newvar();
assert(newvar == 3);

% Edge case 12: variable created by assignin('workspace') that didn't exist before
ws_create_newvar2();
assert(newvar2 == 'created');

% Edge case 13: new variable created by caller, then used in arithmetic
result6 = test_caller_create_and_use();
assert(result6 == 15);

disp('SUCCESS');

% ── Helper functions ──────────────────────────────────────────────────

function val = test_evalin_read()
    val = evalin('workspace', 'x');
end

function test_assignin_write()
    assignin('workspace', 'x', 42);
end

function val = test_evalin_default()
    val = evalin('workspace', 'nonexistent', 99);
end

function test_evalin_error()
    evalin('workspace', 'nonexistent');
end

function test_caller_outer(y)
    caller_reader_y();
    assert(y == 100);
    caller_writer_y();
    assert(y == 555);
end

function val = caller_reader_y()
    val = evalin('caller', 'y');
end

function caller_writer_y()
    assignin('caller', 'y', 555);
end

function result = test_caller_modify()
    z = 0;
    caller_writer_z();
    result = z;
end

function caller_writer_z()
    assignin('caller', 'z', 999);
end

function val = caller_reader_w()
    val = evalin('caller', 'w');
end

function caller_writer_w()
    assignin('caller', 'w', 888);
end

function ws_set_x_vector()
    assignin('workspace', 'x', [2, 5]);
end

function ws_set_x_string()
    assignin('workspace', 'x', 'hello');
end

function ws_set_x_to(val)
    assignin('workspace', 'x', val);
end

function ws_set_x_conditional(flag)
    if flag
        assignin('workspace', 'x', -1);
    end
end

function result = test_caller_type_change()
    m = 0;  % starts as scalar
    caller_set_m_matrix();  % changes to [2,4;6,8]
    result = m;  % must handle the new type
end

function caller_set_m_matrix()
    assignin('caller', 'm', [2, 4; 6, 8]);
end

function result = test_caller_struct_change()
    s = 0;  % starts as scalar
    caller_set_s_struct();  % changes to struct
    result = s;
end

function caller_set_s_struct()
    assignin('caller', 's', struct('value', 42));
end

function result = test_caller_chain()
    v = 0;
    caller_set_v_vector();  % sets v to [10, 20, 30]
    caller_read_v_check();  % reads v via evalin, asserts it's the vector
    result = v;
end

function caller_set_v_vector()
    assignin('caller', 'v', [10, 20, 30]);
end

function caller_read_v_check()
    val = evalin('caller', 'v');
    assert(isequal(val, [10, 20, 30]));
end

function result = test_caller_default_exists()
    q = 7;
    result = caller_read_q_with_default();
end

function val = caller_read_q_with_default()
    val = evalin('caller', 'q', 999);
end

function caller_create_newvar()
    assignin('caller', 'newvar', 3);
end

function ws_create_newvar2()
    assignin('workspace', 'newvar2', 'created');
end

function result = test_caller_create_and_use()
    caller_create_p();
    result = p + 5;
end

function caller_create_p()
    assignin('caller', 'p', 10);
end
