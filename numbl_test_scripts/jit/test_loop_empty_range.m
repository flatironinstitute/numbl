% Test loop JIT with empty range (no-op loop)

% Variables should not be modified when loop doesn't execute
a = 42;
for i = 5:1
    a = 0;
end
assert(a == 42, 'empty range should not modify a');

% Negative step empty range
b = 99;
for k = 1:(-1):5
    b = 0;
end
assert(b == 99, 'empty neg range should not modify b');

% Normal loop after empty range should work
c = 0;
for m = 1:5
    c = c + m;
end
assert(c == 15, 'normal loop after empty should work');
assert(m == 5, 'loop var m should be 5');

disp('SUCCESS')
