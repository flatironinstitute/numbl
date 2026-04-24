% Basic per-assign elemwise correctness — runs under all opt levels and
% must produce the same answer. Under --opt e2 the assigns get compiled
% into per-assign C kernels; under --opt 0 / 1 / e1 they take the regular
% interpreter / JS-JIT / chain-kernel paths. Either way, this script
% checks the numeric output.

n = 5000;
x = linspace(-1, 1, n);
y = exp(x) + sin(x .* 2);
z = y .* x - 0.5;

% Reproducible expected sum (verified in MATLAB R2025b and pinned).
s = sum(z);
assert(abs(s - 1517.667488551138) < 1e-9, ...
    sprintf('elemwise basic: sum(z) = %.16g (expected %.16g)', ...
        s, 1517.667488551138));

disp('SUCCESS')
