% Test string array indexing, assignment, growth, and deletion

arr = ["a" "b"; "c" "d"];

% 2-D and linear reads
assert(arr(2, 1) == "c");
assert(arr(end) == "d");
col2 = arr(:, 2);
assert(isequal(size(col2), [2 1]));
assert(col2(1) == "b" && col2(2) == "d");
lin = arr([1 3]);
assert(isequal(size(lin), [1 2]));
assert(lin(2) == "b");
msk = arr(logical([1 0; 0 1]));
assert(isequal(size(msk), [2 1]));

% brace extraction yields char
s = ["one" "two" "three"];
c = s{2};
assert(ischar(c));
assert(strcmp(c, 'two'));
assert(s{2}(1) == 't');

% scalar string braces
sc = "abc";
assert(ischar(sc{1}));
assert(strcmp(sc{1}, 'abc'));
assert(sc{1}(2) == 'b');
assert(sc(end) == "abc");

% element assignment
la = ["a" "b" "c"];
la(2) = "B";
assert(la(2) == "B");
la(logical([1 0 1])) = ["X" "Z"];
assert(la(1) == "X" && la(2) == "B" && la(3) == "Z");

% scalar expansion
sa = ["a" "b" "c"];
sa(2:3) = "z";
assert(sa(2) == "z" && sa(3) == "z");

% char RHS stays a string array
h2 = ["a" "b"];
h2(2) = 'c';
assert(isstring(h2));
assert(h2(2) == "c");

% 2-D column assignment
ca = ["a" "b"; "c" "d"];
ca(:, 1) = ["p"; "q"];
assert(ca(1, 1) == "p" && ca(2, 1) == "q");

% growth from strings(0) — the accumulate pattern
g = strings(0);
g(end+1) = "x";
g(end+1) = "y";
assert(isequal(size(g), [1 2]));
assert(g(1) == "x" && g(2) == "y");

% growth from a scalar with a gap (values at the gap are unset)
h = "a";
h(4) = "d";
assert(numel(h) == 4);
assert(h(1) == "a" && h(4) == "d");

% deletion
dd = ["a" "b" "c"];
dd(2) = [];
assert(isequal(size(dd), [1 2]));
assert(dd(1) == "a" && dd(2) == "c");

% column vectors keep orientation
cv = ["a"; "b"; "c"];
cv(2) = [];
assert(isequal(size(cv), [2 1]));
cv(end+1) = "z";
assert(isequal(size(cv), [3 1]));

disp('SUCCESS')
