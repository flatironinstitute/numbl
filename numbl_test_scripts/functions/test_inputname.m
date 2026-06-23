% Test inputname: returns the caller's variable name for each argument,
% and '' for arguments that are not plain workspace variables.

% --- Plain variable returns its name ---
x = 5; y = 3;
assert(strcmp(getname1(x, y), 'x'), 'plain variable name');

% --- Literals and expressions have no name ---
assert(strcmp(getname1(5, 3), ''), 'literal has no name');
assert(strcmp(getname1(x + 1, y), ''), 'expression has no name');

% --- Multiple arguments, each a plain variable ---
[n1, n2, n3] = getnames3(x, y, x);
assert(strcmp(n1, 'x') && strcmp(n2, 'y') && strcmp(n3, 'x'), 'multiple names');

% --- Cell indexing: '' for that argument and all subsequent ones ---
c = {1, 2};
[m1, m2, m3] = getnames3(c{2}, x, y);
assert(strcmp(m1, '') && strcmp(m2, '') && strcmp(m3, ''), 'cell-index blanks rest');

% --- Dot indexing: '' for that argument and all subsequent ones ---
s.a = 1;
[p1, p2, p3] = getnames3(x, s.a, y);
assert(strcmp(p1, 'x') && strcmp(p2, '') && strcmp(p3, ''), 'dot-index blanks rest');

% --- Paren indexing: '' for itself but does NOT blank following args ---
v = [10 20 30];
[q1, q2, q3] = getnames3(v(2), x, y);
assert(strcmp(q1, '') && strcmp(q2, 'x') && strcmp(q3, 'y'), 'paren-index is local');

% --- argNumber beyond nargin raises an error ---
err = false;
try
    getOOR(x);
catch
    err = true;
end
assert(err, 'out-of-range argNumber should error');

% --- A nested function reports its own call-site name, not the parent's ---
assert(strcmp(outer(x), 'alpha'), 'nested function call-site name');

disp('SUCCESS');

function s = getname1(a, b)
    s = inputname(1);
end

function [a1, a2, a3] = getnames3(p, q, r)
    a1 = inputname(1);
    a2 = inputname(2);
    a3 = inputname(3);
end

function getOOR(a)
    inputname(2); % nargin == 1, so argument 2 is out of range
end

function r = outer(alpha)
    r = inner(alpha);
    function out = inner(zeta)
        out = inputname(1);
    end
end
