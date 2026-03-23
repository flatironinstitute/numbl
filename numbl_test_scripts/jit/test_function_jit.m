% Test function-level JIT: scalar functions compiled to JS

%!jit
function y = cube(x)
    y = x .* x .* x;
end

a = cube(2);
assert(a == 8);

b = cube(i);
assert(b == -i);

d = cube([i 2 3 4]);
assert(all(d == [-i 8 27 64]));

disp('SUCCESS');
