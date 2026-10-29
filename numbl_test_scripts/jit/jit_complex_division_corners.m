% Complex ./ and .* corner cases in compiled code agree with the
% interpreter and MATLAB (zero elements, infinite divisors, overflow).

z = dv([complex(0, 0) 1i], [0 1]);
assert(isnan(real(z(1))) && imag(z(1)) == 0 && z(2) == 1i);
z = dv([complex(0, 1) 1i], [0 1]);
assert(real(z(1)) == 0 && imag(z(1)) == Inf);
z = dv([complex(1, 2) 1i], [complex(Inf, Inf) 1i]);
assert(isequal(z, [0 1]));
z = dv([3 1], [complex(Inf, Inf) 1i]);
assert(z(1) == 0);
z = dv([complex(1e300, 1e300) 1i], [complex(1e300, 1e300) 1i]);
assert(isequal(z, [1 1]));
z = dv([complex(1e-300, 1e-300) 1i], [complex(1e-300, 1e-300) 1i]);
assert(isequal(z, [1 1]));
z = ml([complex(0, 0) 1i], [Inf 1]);
assert(isnan(real(z(1))) && imag(z(1)) == 0 && z(2) == 1i);
z = ml([complex(0, 1) 1i], [Inf 1]);
assert(real(z(1)) == 0 && imag(z(1)) == Inf);

disp('SUCCESS')

function q = dv(a, b)
    %!numbl:assert_jit c
    q = a ./ b;
end

function q = ml(a, b)
    %!numbl:assert_jit c
    q = a .* b;
end
