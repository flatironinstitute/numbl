% Test that complex values are correctly evaluated as conditions in JIT

function y = complex_if(x)
    if x
        y = 1;
    else
        y = 0;
    end
end

%!jit
assert(complex_if(1 + 2i) == 1);
assert(complex_if(0 + 1i) == 1);
assert(complex_if(1 + 0i) == 1);
assert(complex_if(0 + 0i) == 0);
assert(complex_if(0) == 0);
assert(complex_if(5) == 1);

function y = complex_while(x)
    y = 0;
    while x
        y = y + 1;
        x = x - 1;
    end
end

%!jit
assert(complex_while(1 + 0i) == 1);
assert(complex_while(0 + 0i) == 0);

disp('SUCCESS');
