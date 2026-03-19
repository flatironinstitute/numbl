% Test who() and whos() builtins

% Test 1: who() at script level
x = 5;
y = 'hello';
z = [1 2 3];
C = who();
assert(iscell(C), 'who should return a cell array');
assert(ismember('x', C), 'x should be in who output');
assert(ismember('y', C), 'y should be in who output');
assert(ismember('z', C), 'z should be in who output');

% Test 2: who() with glob pattern
D = who('x*');
assert(length(D) == 1, 'who with x* should match 1 var');
assert(strcmp(D{1}, 'x'), 'x should match x*');

% Test 3: who() with regexp
xx = 10;
E = who('-regexp', '^x');
assert(ismember('x', E), 'x should match ^x');
assert(ismember('xx', E), 'xx should match ^x');
assert(~ismember('y', E), 'y should not match ^x');

% Test 4: who() inside function
test_who_in_function(42, 'abc');

% Test 5: who() excludes undefined variables
test_who_undefined();

% Test 6: whos() returns struct with correct fields
test_whos_struct();

% Test 7: whos() inside function
test_whos_in_function();

% Test 8: whos() with filtering
test_whos_filter();

disp('SUCCESS');

function test_who_in_function(a, b)
    c = 10;
    vars = who();
    assert(ismember('a', vars), 'param a should be in who');
    assert(ismember('b', vars), 'param b should be in who');
    assert(ismember('c', vars), 'local c should be in who');
end

function test_who_undefined()
    x = 1;
    if false
        y = 2;
    end
    vars = who();
    assert(ismember('x', vars), 'x should be in who');
    assert(~ismember('y', vars), 'y should not be in who (never assigned)');
    y = 2;
    vars2 = who();
    assert(ismember('y', vars2), 'y should be in who after assignment');
end

function test_whos_struct()
    a = 42;
    b = 'text';
    c = [1 2; 3 4];
    S = whos();
    assert(strcmp(S(1).name, 'a'), 'first var should be a');
    assert(strcmp(S(1).class, 'double'), 'a should be double');
    assert(isequal(S(1).size, [1 1]), 'a should be 1x1');
    assert(S(1).bytes == 8, 'a should be 8 bytes');
    assert(strcmp(S(2).name, 'b'), 'second var should be b');
    assert(strcmp(S(2).class, 'char'), 'b should be char');
    assert(strcmp(S(3).name, 'c'), 'third var should be c');
    assert(isequal(S(3).size, [2 2]), 'c should be 2x2');
    assert(S(3).bytes == 32, 'c should be 32 bytes');
end

function test_whos_in_function()
    x = 5;
    y = 'hello';
    S = whos();
    names = {};
    for k = 1:length(S)
        names{k} = S(k).name;
    end
    assert(ismember('x', names), 'x should be in whos');
    assert(ismember('y', names), 'y should be in whos');
end

function test_whos_filter()
    alpha = 1;
    beta = 2;
    S = whos('a*');
    assert(length(S) == 1, 'whos with a* should match 1 var');
    assert(strcmp(S.name, 'alpha'), 'should match alpha');
end
