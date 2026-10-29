% Fused element-wise gpuArray expressions (one kernel per chain on a device
% that fuses) agree with the host: long chains past the kernel's limits, a
% buffer read several times, lazy results read by reductions, indexing and
% conditions, values across calls and in containers, and memory release.

rng(4);
A = single(randn(300, 200)); B = single(randn(300, 200));
GA = gpuArray(A); GB = gpuArray(B);
chk = @(g, h, tol) assert(max(abs(double(gather(g(:))) - double(h(:)))) <= tol * max(1, max(abs(double(h(:))))), 'mismatch');
% long chain exceeding node/input/scalar limits
G = GA; H = A;
for k = 1:30, G = G * 0.9 + GB .* 0.01 - k / 100; H = H * 0.9 + B .* 0.01 - k / 100; end
chk(G, H, 1e-5);
% one statement with many terms
Y = sin(GA) .* cos(GB) + abs(GA - GB) ./ (1 + GA .* GA) - exp(-GB .* GB) + (GA > 0) .* 2 + max(GA, GB) - min(GA, 0.5);
Yh = sin(A) .* cos(B) + abs(A - B) ./ (1 + A .* A) - exp(-B .* B) + (A > 0) .* 2 + max(A, B) - min(A, 0.5);
% (WGSL allows an absolute 2^-11 for sin and cos)
chk(Y, Yh, 1e-3);
% same buffer many times
Z = GA + GA .* GA - GA ./ (GA + 2); Zh = A + A .* A - A ./ (A + 2); chk(Z, Zh, 1e-5);
% lazy consumed by reductions, indexing, gather, comparisons in if
chk(sum(GA .* GB), sum(A .* B), 1e-5);
chk(max(GA + GB, [], 2), max(A + B, [], 2), 0);
W = GA .* 2; chk(W(3, :), A(3, :) * 2, 0);
if any(GA(:) .* 0 + 1 > 0.5), ok = 1; else ok = 0; end; assert(ok == 1);
% across function calls and in a cell / struct
F = scale(GA, GB); chk(F, (A + B) * 3 - 1, 1e-6);
c = {GA + 1, GB .* 2}; s.v = GA - GB;
chk(c{1}, A + 1, 0); chk(c{2}, B * 2, 0); chk(s.v, A - B, 0);
% overwrite an input in the same statement
Q = GA; Q = Q .* 2 + Q; chk(Q, A * 3, 1e-6);
% logical results and mixed
L = (GA > 0) & (GB < 0); assert(strcmp(underlyingType(L), 'logical')); assert(isequal(gather(L), (A > 0) & (B < 0)));
% memory returns
d = gpuDevice; used0 = d.TotalMemory - d.AvailableMemory;
for k = 1:20, T = GA .* k + GB; end
clear T; d = gpuDevice; assert(d.TotalMemory - d.AvailableMemory <= used0, 'leak');
disp('SUCCESS')
function r = scale(a, b)
  t = a + b;
  r = t * 3 - 1;
end
