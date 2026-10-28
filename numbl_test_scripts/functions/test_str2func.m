% str2func: handles from names and anonymous functions from text, and the
% func2str round trip.

% named functions
f = str2func('sin');
assert(isa(f, 'function_handle'));
assert(f(0) == 0);
assert(strcmp(func2str(f), 'sin'));
assert(ischar(func2str(f)));
g = str2func("cos");
assert(g(0) == 1);
h = str2func('@sin');
assert(strcmp(func2str(h), 'sin'));
p = str2func('plus');
assert(p(1, 2) == 3);

% local functions are reachable by name
k = str2func('str2func_local');
assert(k(3) == 30);
assert(strcmp(func2str(k), 'str2func_local'));

% a handle to an unknown name is created; calling it errors
n = str2func('str2func_no_such_fn');
assert(strcmp(func2str(n), 'str2func_no_such_fn'));
try
    n(1);
    error('expected an error');
catch e
    assert(strcmp(e.identifier, 'MATLAB:UndefinedFunction'));
end

% anonymous functions from text
a1 = str2func('@(x) x+1');
assert(a1(2) == 3);
assert(strcmp(func2str(a1), '@(x)x+1'));
a2 = str2func('@(x,y) x*y');
assert(a2(2, 3) == 6);
assert(nargin(a2) == 2);
a3 = str2func('@() 42');
assert(a3() == 42);
a4 = str2func('@(varargin) numel(varargin)');
assert(a4(1, 2, 3) == 3);

% the body is evaluated in an empty workspace: no variables, no local
% functions
a = 5;
b1 = str2func('@(x) x+a');
try
    b1(1);
    error('expected an error');
catch e
    assert(strcmp(e.identifier, 'MATLAB:UndefinedFunction'));
end
b2 = str2func('@(x) str2func_local(x)');
try
    b2(1);
    error('expected an error');
catch e
    assert(strcmp(e.identifier, 'MATLAB:UndefinedFunction'));
end
assert(str2func_in_function() == -1);

% func2str / str2func round trip
c = 3;
fns = {@(y) y.^2 + 1, @(x, y) x*y - c, @() [1 2; 3 4], @(s) s.a(2), ...
       @(x) -x.^2, @(x) (-x).^2, @(x) 2.^-x, @(x) (x + 1)*2, @(v) v(end), ...
       @(x) x', @(x) {x, 'it''s'}, @(x) x(1:2:end), @(m) m(:, 1), ...
       @(x) ~(x > 1) && true, @(x) 3i*x, @(x) @(y) x + y};
txt = {'@(y)y.^2+1', '@(x,y)x*y-c', '@()[1,2;3,4]', '@(s)s.a(2)', ...
       '@(x)-x.^2', '@(x)(-x).^2', '@(x)2.^-x', '@(x)(x+1)*2', '@(v)v(end)', ...
       '@(x)x''', '@(x){x,''it''''s''}', '@(x)x(1:2:end)', '@(m)m(:,1)', ...
       '@(x)~(x>1)&&true', '@(x)3i*x', '@(x)@(y)x+y'};
for i = 1:numel(fns)
    s = func2str(fns{i});
    assert(strcmp(s, txt{i}), sprintf('func2str: got %s, expected %s', s, txt{i}));
end
r = str2func(func2str(@(y) y.^2 + 1));
assert(r(3) == 10);
r = str2func(func2str(@(x) (-x).^2));
assert(r(2) == 4);
r = str2func(func2str(@(x) 2.^-x));
assert(r(1) == 0.5);

% invalid input
try
    str2func(3);
    error('expected an error');
catch e
    assert(strcmp(e.identifier, 'MATLAB:string:MustBeStringScalarOrCharacterVector'));
    assert(strcmp(e.message, 'Argument must be a text scalar.'));
end

disp('SUCCESS');

function y = str2func_local(x)
    y = x * 10;
end

function r = str2func_in_function()
    q = 7; %#ok<NASGU>
    f = str2func('@(x) x + q');
    try
        r = f(1);
    catch
        r = -1;
    end
end
