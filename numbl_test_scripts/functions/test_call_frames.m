% Call frames: evalin/assignin('caller') from nested code, and inputname in
% functions that a builtin calls back.

% A nested evalin('caller') inside evalin('caller', ...) refers to the same
% caller.
v = 'base';
assert(strcmp(frame_mid(), 'mid'));

% In an anonymous function body, 'caller' is the function that called the
% anonymous function.
w = 'top';
assert(strcmp(anon_caller(), 'anon_caller'));
assert(strcmp(anon_assign(), 'assigned'));

% A function that a builtin calls back sees the argument names of the
% builtin's call expression, on every callback.
x = 5;
h = @name_of_first;
c = cellfun(h, {x, x}, 'UniformOutput', false);
assert(strcmp(c{1}, 'h') && strcmp(c{2}, 'h'));

disp('SUCCESS');

function r = frame_mid()
    v = 'mid'; %#ok<NASGU>
    r = frame_inner();
end

function r = frame_inner()
    v = 'inner'; %#ok<NASGU>
    r = evalin('caller', 'evalin(''caller'', ''v'')');
end

function s = name_of_first(a) %#ok<INUSD>
    s = inputname(1);
end

function r = anon_caller()
    w = 'anon_caller'; %#ok<NASGU>
    f = @() evalin('caller', 'w');
    r = f();
end

function r = anon_assign()
    r = 'unset';
    f = @() assignin('caller', 'r', 'assigned');
    f();
end
