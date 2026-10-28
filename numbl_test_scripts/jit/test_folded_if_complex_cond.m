% An if whose condition folds at compile time (exact complex operand)
% still compiles: the kept condition does not bind ans.
%!numbl:assert_jit c
x = sqrt(-4);
if ~isfinite(real(x))
    error('sqrt(-4) real part not finite');
end
y = exp(1i * pi / 2);
if abs(imag(y) - 1) > 1e-12
    error('exp(i*pi/2) imag=%g', imag(y));
end
z = log(-1e200);
if abs(real(z) - log(1e200)) > 1e-6
    error('log(-1e200) re=%g', real(z));
end
if imag(x) ~= 2
    error('sqrt(-4) imag=%g', imag(x));
end
disp('SUCCESS');
