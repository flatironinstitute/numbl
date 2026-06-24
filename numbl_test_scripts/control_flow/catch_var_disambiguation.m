% `catch <ident>` captures the exception only when the identifier stands alone
% on the line (followed by a separator or `end`). `catch ident = ...` instead
% begins the catch body with an assignment statement -- the identifier is a
% normal variable, not the exception object. (rktoolbox sym2rkfun.m uses the
% single-line `try ...; catch v = []; end` form.)

% --- assignment in the catch body on the same line ---
try
    error('boom');
catch y = [];
end
assert(~islogical(y) && isempty(y), 'catch y = [] should run as a statement');
assert(isa(y, 'double'), 'y should be a double []');

% --- single-line try/catch/end, the sym2rkfun pattern ---
try   z = 1 + 1;   catch z = [];  end
assert(z == 2, 'single-line try body ran');

try   error('x');   catch z = -5;  end
assert(z == -5, 'single-line catch assignment ran');

% --- `catch e` alone still captures the exception object ---
try
    error('myid:sub', 'detail %d', 9);
catch e
    assert(~isempty(strfind(e.message, '9')), 'exception message captured');
    assert(~isempty(strfind(e.identifier, 'myid:sub')), 'exception id captured');
end

% --- `catch ME,` with a trailing comma still captures the exception ---
try
    error('another');
catch ME, assert(~isempty(strfind(ME.message, 'another')), 'comma-form capture');
end

% --- `catch` with a function-call statement (not an identifier capture) ---
flag = 0;
try
    error('z');
catch
    flag = mark(flag);
end
assert(flag == 1, 'catch body statement ran');

disp('SUCCESS');

function out = mark(in)
    out = in + 1;
end
