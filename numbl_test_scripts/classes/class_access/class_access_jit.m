% Access checks hold in code the JIT would otherwise compile (loops and
% functions over a simple value class).

v = AttrValue_();
s = 0;
for i = 1:20
    s = s + v.y + v.getX();
end
assert(s == 60);
assert(sumPublic(v, 20) == 40);
caught = false;
try
    readPrivate(v, 20);
catch e
    caught = strcmp(e.identifier, 'MATLAB:class:GetProhibited');
end
assert(caught);

disp('SUCCESS');

function s = sumPublic(v, n)
s = 0;
for i = 1:n
    s = s + v.y;
end
end

function s = readPrivate(v, n)
s = 0;
for i = 1:n
    s = s + v.x;
end
end
