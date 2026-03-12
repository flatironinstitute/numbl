% Test cell array indexing with colon, ranges, and comma-separated list expansion
%
% Key MATLAB distinction:
%   X(1:2)  — parenthesis indexing → returns a new cell array (subset)
%   X{1:2}  — curly-brace indexing → returns a comma-separated list (multiple outputs)
%   X{:}    — curly-brace colon   → CSL of all elements

%% ---- Curly-brace colon: X{:} produces CSL ----

%% Basic: [a, b, c] = X{:} with numbers
X = {1, 2, 3};
[a, b, c] = X{:};
assert(a == 1);
assert(b == 2);
assert(c == 3);

%% Mixed types
Y = {10, 'hello', [4 5 6]};
[p, q, r] = Y{:};
assert(p == 10);
assert(strcmp(q, 'hello'));
assert(isequal(r, [4 5 6]));

%% Single element cell
S = {42};
[v] = S{:};
assert(v == 42);

%% Two element cell
T = {'alpha', 'beta'};
[t1, t2] = T{:};
assert(strcmp(t1, 'alpha'));
assert(strcmp(t2, 'beta'));

%% Cell containing nested cells
N = {{1, 2}, {3, 4}};
[n1, n2] = N{:};
assert(iscell(n1));
assert(iscell(n2));
assert(n1{1} == 1);
assert(n2{2} == 4);

%% Cell containing matrices
M = {[1 2; 3 4], [5 6; 7 8]};
[m1, m2] = M{:};
assert(isequal(m1, [1 2; 3 4]));
assert(isequal(m2, [5 6; 7 8]));

%% ---- Curly-brace range: X{1:2} produces CSL ----

R = {10, 20, 30, 40};
[r1, r2] = R{1:2};
assert(r1 == 10);
assert(r2 == 20);

%% Range indexing: X{2:4}
[r2b, r3, r4] = R{2:4};
assert(r2b == 20);
assert(r3 == 30);
assert(r4 == 40);

%% end keyword in curly-brace indexing
E = {'a', 'b', 'c', 'd'};
assert(strcmp(E{end}, 'd'));

%% Range with end: X{2:end}
[e2, e3, e4] = E{2:end};
assert(strcmp(e2, 'b'));
assert(strcmp(e3, 'c'));
assert(strcmp(e4, 'd'));

%% ---- Parenthesis indexing: X(1:2) returns a sub-cell ----

P = {100, 'two', [3 3 3], 'four'};
P2 = P(1:2);
assert(iscell(P2));
assert(numel(P2) == 2);
assert(P2{1} == 100);
assert(strcmp(P2{2}, 'two'));

%% Parenthesis colon: X(:) returns the entire cell
P3 = P(:);
assert(iscell(P3));
assert(numel(P3) == 4);

%% Parenthesis with end
P4 = P(2:end);
assert(iscell(P4));
assert(numel(P4) == 3);
assert(strcmp(P4{1}, 'two'));

%% ---- CSL expansion in function call arguments: foo(X{:}) ----

%% Basic: pass cell contents as separate arguments
function check3(a, b, c)
    assert(a == 1);
    assert(b == 2);
    assert(c == 3);
end
F = {1, 2, 3};
check3(F{:});

%% Mixed types in function args
function check_mixed(x, y, z)
    assert(x == 10);
    assert(strcmp(y, 'hello'));
    assert(isequal(z, [4 5 6]));
end
G = {10, 'hello', [4 5 6]};
check_mixed(G{:});

%% CSL with range in function args: foo(X{1:2})
function check2(a, b)
    assert(a == 100);
    assert(b == 200);
end
H = {100, 200, 300};
check2(H{1:2});

%% CSL mixed with regular args: foo(1, X{:}, 5)
function check5(a, b, c, d, e)
    assert(a == 1);
    assert(b == 10);
    assert(c == 20);
    assert(d == 30);
    assert(e == 5);
end
J = {10, 20, 30};
check5(1, J{:}, 5);

%% CSL with varargin forwarding: cell{:} → varargin → forwarded as varargin{:}
function forward_va(varargin)
    check_va(8, varargin{:})
end
function check_va(a, b, c, d)
    assert(a == 8);
    assert(b == 1);
    assert(c == 2);
    assert(d == 3);
end
K = {1, 2, 3};
forward_va(K{:});

%% ---- Scalar curly-brace indexing (regression check) ----

C = {100, 200, 300};
assert(C{1} == 100);
assert(C{2} == 200);
assert(C{3} == 300);

disp('SUCCESS')
