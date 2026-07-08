% Test string array operators: ==, ~=, +, <, isequal

% elementwise equality against a scalar
eqr = ["a" "b"] == "a";
assert(islogical(eqr));
assert(isequal(size(eqr), [1 2]));
assert(eqr(1) && ~eqr(2));
ner = ["a" "b"] ~= "a";
assert(~ner(1) && ner(2));

% array vs array
both = ["a" "b"] == ["a" "z"];
assert(both(1) && ~both(2));

% + appends elementwise
pl = ["a" "b"] + "x";
assert(isstring(pl));
assert(pl(1) == "ax" && pl(2) == "bx");
p2 = "pre" + ["a" "b"];
assert(p2(1) == "prea" && p2(2) == "preb");
p3 = ["a" "b"] + ["1" "2"];
assert(p3(1) == "a1" && p3(2) == "b2");

% + with a number appends its text form
p4 = "abc" + 1;
assert(isstring(p4));
assert(p4 == "abc1");
p5 = "ab" + 'c';
assert(p5 == "abc");

% lexicographic comparison
assert("abc" < "abd");
assert("b" > "a");

% strcmp over arrays
scr = strcmp(["a" "b"], "a");
assert(isequal(size(scr), [1 2]));
assert(scr(1) && ~scr(2));

% contains / startsWith / endsWith over arrays
cw = contains(["abc" "xyz"], "b");
assert(islogical(cw));
assert(cw(1) && ~cw(2));
sw = startsWith(["abc" "xyz"], "a");
assert(sw(1) && ~sw(2));
ew = endsWith(["abc" "xyz"], "z");
assert(~ew(1) && ew(2));

% isequal
assert(isequal(["a" "b"], ["a" "b"]));
assert(~isequal(["a" "b"], ["a" "c"]));
assert(~isequal(["a" "b"], ["a"; "b"]));
assert(isequal("a", 'a'));

% switch on string scalars still works
x = "foo";
hit = false;
switch x
    case "foo"
        hit = true;
end
assert(hit);

disp('SUCCESS')
