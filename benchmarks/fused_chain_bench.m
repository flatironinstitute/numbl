% Element-wise chains on large vectors. Each kernel is one MATLAB
% statement that lowers to several ANF temps; at --opt 2 the inline
% pass folds them into one fused C loop with no intermediate tensors.
% A script (not a function) so the loop/top-level C-JIT executors run it.
N = 2000000;
trials = 50;
t = linspace(-1.0, 1.0, N)';
a = sin(3.1 .* t);
b = cos(2.7 .* t + 0.4);
c = t .* t;
d = 1 - t;
e = 0.5 .* t;
y = a;
v = a;
w = a;

tic;
for k = 1:trials
    y = a .* b + c .* d - e;
end
t1 = toc;
tic;
for k = 1:trials
    v = v + 0.01 .* (a .* v - b);
end
t2 = toc;
tic;
for k = 1:trials
    w = exp(-a .* a) .* cos(b) + c;
end
t3 = toc;

fprintf('N=%d, trials=%d\n', N, trials);
fprintf('Arith  y = a.*b + c.*d - e:          %7.3f s\n', t1);
fprintf('Update v = v + 0.01 .* (a.*v - b):   %7.3f s\n', t2);
fprintf('Unary  w = exp(-a.*a) .* cos(b) + c: %7.3f s\n', t3);
fprintf('total = %.3f s\n', t1 + t2 + t3);
fprintf('check = %.10g\n', sum(y) + sum(v) + sum(w));
