% Complex division corner cases (zero, Inf, NaN, overflow) in scalar and
% array form. The second element keeps each array complex.

% 0/0 is NaN, also inside a complex array
assert(isnan(complex(0, 0) / complex(0, 0)));
z = [complex(0, 0) 1i] ./ [complex(0, 0) 1i];
assert(isnan(z(1)) && z(2) == 1 && isreal(z));
z = [complex(0, 0) 1i] ./ [0 1];
assert(isnan(real(z(1))) && imag(z(1)) == 0 && z(2) == 1i);
z = [complex(0, 0) 1] ./ [complex(0, 0) 1i];
assert(isnan(real(z(1))) && imag(z(1)) == 0 && z(2) == -1i);
z = [0 1] ./ [complex(0, 0) 1i];
assert(isnan(real(z(1))) && imag(z(1)) == 0);
z = [complex(0, 0) 1i] ./ [NaN 1];
assert(isnan(real(z(1))) && imag(z(1)) == 0);
z = [complex(0, 0) 1i] .\ [0 1];
assert(isnan(real(z(1))) && imag(z(1)) == 0);

% The same rule for times: a zero element times Inf is NaN
z = [complex(0, 0) 1i] .* [Inf 1];
assert(isnan(real(z(1))) && imag(z(1)) == 0 && z(2) == 1i);
z = [Inf 1] .* [complex(0, 0) 1i];
assert(isnan(real(z(1))) && imag(z(1)) == 0);
z = [complex(0, 1) 1i] .* [Inf 1];
assert(real(z(1)) == 0 && imag(z(1)) == Inf);

% Nonzero over zero: a signed Inf per part, a zero part stays 0
z = [complex(1, 2) 1] ./ [complex(0, 0) 1i];
assert(isequal(z(1), complex(Inf, Inf)));
z = [complex(0, 1) 1i] ./ [0 1];
assert(real(z(1)) == 0 && imag(z(1)) == Inf);
z = [complex(0, -1) 1i] ./ [complex(0, 0) 1];
assert(real(z(1)) == 0 && imag(z(1)) == -Inf);
z = [complex(0, 1) 1i] ./ [NaN 1];
assert(real(z(1)) == 0 && isnan(imag(z(1))));
z = complex(1, 2) / 0;
assert(real(z) == Inf && imag(z) == Inf);

% Infinite divisors
assert(complex(1, 2) / complex(Inf, Inf) == 0);
assert(complex(1, 2) / complex(0, Inf) == 0);
assert(3 / complex(Inf, 1) == 0);
z = [complex(1, 2) 1i] ./ [complex(Inf, Inf) 1i];
assert(isequal(z, [0 1]));
z = [3 1] ./ [complex(Inf, Inf) 1i];
assert(z(1) == 0);
z = [complex(Inf, Inf) 1i] ./ [complex(Inf, Inf) 1i];
assert(isnan(real(z(1))) && isnan(imag(z(1))));
z = [complex(Inf, 1) 1i] ./ [complex(Inf, 1) 1i];
assert(isnan(real(z(1))) && isnan(imag(z(1))));
z = [Inf 1] ./ [complex(1, 1) 1i];
assert(real(z(1)) == Inf && imag(z(1)) == -Inf);
z = [complex(Inf, 0) 1i] ./ [complex(1, 1) 1i];
assert(real(z(1)) == Inf && imag(z(1)) == -Inf);

% No spurious overflow or underflow
assert(complex(1e300, 1e300) / complex(1e300, 1e300) == 1);
z = [complex(1e300, 1e300) 1i] ./ [complex(1e300, 1e300) 1i];
assert(isequal(z, [1 1]));
z = [complex(1e-300, 1e-300) 1i] ./ [complex(1e-300, 1e-300) 1i];
assert(isequal(z, [1 1]));
z = [complex(1e308, 1e308) 1i] ./ [complex(1e-308, 1e-308) 1i];
assert(real(z(1)) == Inf && imag(z(1)) == 0);

% Ordinary values
z = [complex(1, 2) 1i] ./ [complex(3, 4) 1i];
assert(abs(z(1) - complex(0.44, 0.08)) < 1e-15);
w = complex(1, 2) / complex(3, 4);
assert(abs(w - complex(0.44, 0.08)) < 1e-15);

disp('SUCCESS');
