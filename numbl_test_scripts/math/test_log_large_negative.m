% log / log2 / log10 of a large-magnitude negative must not overflow
x = log(-1e200);
if ~isfinite(real(x))
    error('log(-1e200) re not finite: %g', real(x));
end
if abs(real(x) - log(1e200)) > 1e-6
    error('log(-1e200) re=%g, expected %g', real(x), log(1e200));
end
if abs(imag(x) - pi) > 1e-12
    error('log(-1e200) im=%g, expected pi', imag(x));
end

y = log2(-1e200);
if ~isfinite(real(y))
    error('log2(-1e200) re not finite: %g', real(y));
end
if abs(real(y) - log2(1e200)) > 1e-6
    error('log2(-1e200) re=%g, expected %g', real(y), log2(1e200));
end

z = log10(-1e200);
if ~isfinite(real(z))
    error('log10(-1e200) re not finite: %g', real(z));
end
if abs(real(z) - 200) > 1e-10
    error('log10(-1e200) re=%g, expected 200', real(z));
end

disp('SUCCESS');
