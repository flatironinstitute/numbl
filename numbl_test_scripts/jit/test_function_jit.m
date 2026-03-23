% Test function-level JIT: scalar functions compiled to JS

function y = cube(x)
    y = x .* x .* x;
end

%!jit
a = cube(2);
assert(a == 8);

%!jit
b = cube(i);
assert(b == -i);

%!jit
d = cube([i 2 3 4]);
assert(all(d == [-i 8 27 64]));

disp('SUCCESS');
