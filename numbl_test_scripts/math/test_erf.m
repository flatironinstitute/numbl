% Test erf, erfc, erfinv, erfcinv

% erf basics
assert(abs(erf(0.76) - 0.71753675280559082) < 1e-10, 'erf(0.76)');
assert(abs(erf(0) - 0) < 1e-15, 'erf(0)');
assert(abs(erf(1) - 0.84270079294971489) < 1e-14, 'erf(1)');
assert(abs(erf(-1) + erf(1)) < 1e-15, 'erf odd symmetry');
assert(abs(erf(-0.5) + erf(0.5)) < 1e-15, 'erf odd symmetry 2');
assert(abs(erf(6) - 1) < 1e-15, 'erf(6) ~ 1');
assert(abs(erf(-6) + 1) < 1e-15, 'erf(-6) ~ -1');

% erfc basics
assert(abs(erfc(0) - 1) < 1e-15, 'erfc(0)');
assert(abs(erfc(1) - (1 - erf(1))) < 1e-14, 'erfc(1)');
assert(abs(erfc(0.5) - (1 - erf(0.5))) < 1e-14, 'erfc(0.5)');
assert(abs(erfc(-1) - (1 - erf(-1))) < 1e-14, 'erfc(-1)');

% erfc in the [1.25, 6] range
assert(abs(erfc(1.5) - 0.03389485352468927) < 1e-13, 'erfc(1.5)');
assert(abs(erfc(1.8214) - 0.00999944247630204) < 1e-13, 'erfc(1.8214)');
assert(abs(erfc(2.0) - 0.00467773498104727) < 1e-13, 'erfc(2.0)');
assert(erfc(30) == 0, 'erfc(30) = 0');
assert(abs(erfc(-6) - 2) < 1e-15, 'erfc(-6) ~ 2');

% erf in [1.25, 6] range
assert(abs(erf(1.5) - 0.96610514647531073) < 1e-13, 'erf(1.5)');
assert(abs(erf(2.0) - 0.99532226501895273) < 1e-13, 'erf(2.0)');

% erf vector input
V = [-0.5 0 1 0.72];
R = erf(V);
assert(abs(R(1) - (-0.52049987781304652)) < 1e-12, 'erf vector(1)');
assert(abs(R(2)) < 1e-15, 'erf vector(2)');
assert(abs(R(3) - 0.84270079294971489) < 1e-12, 'erf vector(3)');

% erf matrix input
M = [0.29 -0.11; 3.1 -2.9];
R = erf(M);
assert(abs(R(1,1) - 0.3183) < 1e-3, 'erf matrix(1,1)');
assert(abs(R(1,2) - (-0.1236)) < 1e-3, 'erf matrix(1,2)');
assert(abs(R(2,1) - 1.0) < 1e-4, 'erf matrix(2,1)');
assert(abs(R(2,2) - (-1.0)) < 1e-4, 'erf matrix(2,2)');

% erfinv basics
assert(erfinv(0) == 0, 'erfinv(0)');
assert(erfinv(1) == Inf, 'erfinv(1)');
assert(erfinv(-1) == -Inf, 'erfinv(-1)');
assert(isnan(erfinv(1.5)), 'erfinv out of range');

% erfinv roundtrip: erf(erfinv(x)) == x
vals = [-0.9, -0.5, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99];
for i = 1:length(vals)
    assert(abs(erf(erfinv(vals(i))) - vals(i)) < 1e-12, 'erfinv roundtrip');
end

% erfinv reverse roundtrip: erfinv(erf(x)) == x
xvals = [-2, -1, -0.5, 0, 0.5, 1, 2];
for i = 1:length(xvals)
    assert(abs(erfinv(erf(xvals(i))) - xvals(i)) < 1e-10, 'erfinv reverse roundtrip');
end

% erfinv symmetry
assert(abs(erfinv(-0.5) + erfinv(0.5)) < 1e-15, 'erfinv odd');

% erfinv vector
R = erfinv([0 0.5 -0.5]);
assert(R(1) == 0);
assert(abs(R(2) + R(3)) < 1e-15);

% erfcinv basics
assert(erfcinv(1) == 0, 'erfcinv(1)');
assert(erfcinv(0) == Inf, 'erfcinv(0)');
assert(erfcinv(2) == -Inf, 'erfcinv(2)');
assert(isnan(erfcinv(-0.1)), 'erfcinv out of range');

% erfcinv roundtrip: erfc(erfcinv(x)) == x
cvals = [0.01, 0.1, 0.5, 1.0, 1.5, 1.9, 1.99];
for i = 1:length(cvals)
    assert(abs(erfc(erfcinv(cvals(i))) - cvals(i)) < 1e-12, 'erfcinv roundtrip');
end

% erfcinv(x) = erfinv(1-x)
assert(abs(erfcinv(0.3) - erfinv(0.7)) < 1e-14, 'erfcinv = erfinv(1-x)');

% erfcx basics: erfcx(x) = exp(x^2) * erfc(x)
assert(abs(erfcx(0) - 1) < 1e-14, 'erfcx(0)');
assert(abs(erfcx(0.5) - exp(0.25)*erfc(0.5)) < 1e-12, 'erfcx(0.5)');
assert(abs(erfcx(1.0) - exp(1)*erfc(1.0)) < 1e-12, 'erfcx(1.0)');
assert(abs(erfcx(2.0) - exp(4)*erfc(2.0)) < 1e-10, 'erfcx(2.0)');
assert(abs(erfcx(-1.0) - exp(1)*erfc(-1.0)) < 1e-12, 'erfcx(-1.0)');

% erfcx large x: approaches 1/(x*sqrt(pi))
assert(abs(erfcx(100) - 1/(100*sqrt(pi)))/(1/(100*sqrt(pi))) < 0.001, 'erfcx(100)');

% erfcx special values
assert(erfcx(Inf) == 0, 'erfcx(Inf)');
assert(erfcx(-Inf) == Inf, 'erfcx(-Inf)');
assert(isnan(erfcx(NaN)), 'erfcx(NaN)');

% CDF of normal distribution
x = 1.5;
cdf = (1/2)*(1 + erf(x/sqrt(2)));
assert(abs(cdf - 0.9331928) < 1e-6, 'normal CDF');

disp('SUCCESS');
