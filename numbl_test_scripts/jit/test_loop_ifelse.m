% Test loop JIT with if/elseif/else inside loop body
a = 0;
b = 0;
c = 0;
for i = 1:30
    if mod(i, 3) == 0
        a = a + i;
    elseif mod(i, 3) == 1
        b = b + i;
    else
        c = c + i;
    end
end
assert(a == 165, 'divisible by 3 sum wrong');
assert(b == 145, 'mod 1 sum wrong');
assert(c == 155, 'mod 2 sum wrong');

disp('SUCCESS')
