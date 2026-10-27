% Local functions take precedence over builtins of the same name, including
% builtins that need the caller's scope and constants.
assert(exist(1) == 101);
assert(strcmp(which('x'), 'local which'));
assert(isa(3, 'double') == 7);
assert(pi == 3);
assert(pi() == 3);
disp('SUCCESS')

function r = exist(x)
r = 100 + x;
end

function r = which(~)
r = 'local which';
end

function r = isa(~, ~)
r = 7;
end

function r = pi()
r = 3;
end
