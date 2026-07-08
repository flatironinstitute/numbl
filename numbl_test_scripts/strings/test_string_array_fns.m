% Test string-array functions: split/join/unique/sort/cellstr/char/double/...

% the MATLAB doc example: split and find unique words
str = "A horse! A horse! My kingdom for a horse!";
str = erase(str, "!");
str = lower(str);
str = split(str);
assert(isstring(str));
assert(isequal(size(str), [9 1]));
assert(str(1) == "a" && str(2) == "horse");
u = unique(str);
assert(isequal(size(u), [5 1]));
assert(u(1) == "a" && u(3) == "horse" && u(5) == "my");

% split with a delimiter
p = split("a,b,c", ",");
assert(isequal(size(p), [3 1]));
assert(p(2) == "b");

% join
j = join(["a" "b" "c"]);
assert(isequal(size(j), [1 1]));
assert(j == "a b c");
j2 = join(["a" "b"], "-");
assert(j2 == "a-b");

% sort and unique keep vector orientation
so = sort(["b" "a"]);
assert(isequal(size(so), [1 2]));
assert(so(1) == "a" && so(2) == "b");
ur = unique(["b" "a" "b"]);
assert(isequal(size(ur), [1 2]));
assert(ur(1) == "a");

% strjoin accepts string arrays and returns a string
sj = strjoin(["a" "b"], ",");
assert(isstring(sj));
assert(sj == "a,b");

% cellstr
cs = cellstr(["a" "bb"]);
assert(iscell(cs));
assert(isequal(size(cs), [1 2]));
assert(strcmp(cs{2}, 'bb'));

% char of a string column pads rows
ch = char(["ab"; "c"]);
assert(ischar(ch));
assert(isequal(size(ch), [2 2]));
assert(strcmp(ch(1, :), 'ab'));
assert(strcmp(ch(2, :), 'c '));

% double parses text; str2double too
d = double(["256" "3.1416" "8.9e-3"]);
assert(abs(d(1) - 256) < 1e-12);
assert(abs(d(2) - 3.1416) < 1e-12);
assert(abs(d(3) - 0.0089) < 1e-12);
dn = double(["1" "abc"]);
assert(dn(1) == 1 && isnan(dn(2)));
y = str2double(["2.5" "nope"]);
assert(y(1) == 2.5 && isnan(y(2)));
z = double("42");
assert(z == 42);

% strlength is elementwise
sl = strlength(["ab" "c"]);
assert(isequal(size(sl), [1 2]));
assert(sl(1) == 2 && sl(2) == 1);

% lower/upper/replace map elementwise and stay string arrays
lo = lower(["AB" "Cd"]);
assert(isstring(lo));
assert(lo(1) == "ab" && lo(2) == "cd");
rp = replace(["a-b" "c-d"], "-", "+");
assert(rp(1) == "a+b" && rp(2) == "c+d");
er = erase(["a-b" "c-d"], "-");
assert(er(1) == "ab" && er(2) == "cd");

% string() conversions
sc = string({'a', 'bb'; 'c', 'dd'});
assert(isstring(sc));
assert(isequal(size(sc), [2 2]));
assert(sc(2, 2) == "dd");
scm = string(['ab'; 'cd']);
assert(isequal(size(scm), [2 1]));
assert(scm(2) == "cd");
snm = string([1 2; 3 4]);
assert(isequal(size(snm), [2 2]));
assert(snm(2, 1) == "3");
assert(string(pi) == "3.1416");
assert(string(true) == "true");

% strsplit on a string returns a string array
ss = strsplit("a b");
assert(isstring(ss));
assert(numel(ss) == 2);
assert(ss(2) == "b");

% ismember
im = ismember(["a" "z"], ["a" "b"]);
assert(im(1) && ~im(2));

% for iterates columns
n = 0;
for t = ["a" "b" "c"]
    n = n + 1;
    last = t;
end
assert(n == 3);
assert(last == "c");
n2 = 0;
for t = ["a" "b"; "c" "d"]
    n2 = n2 + 1;
    sz = size(t);
end
assert(n2 == 2);
assert(isequal(sz, [2 1]));

% sprintf consumes string-array elements
sp = sprintf('%s|', ["a" "b"]);
assert(strcmp(sp, 'a|b|'));

% ismissing on ordinary strings is all-false
mm = ismissing(["a" ""]);
assert(islogical(mm));
assert(~mm(1) && ~mm(2));

disp('SUCCESS')
