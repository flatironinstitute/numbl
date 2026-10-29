% gpuArray: creation, gather, type queries, operations that stay on the
% device, indexing and concatenation. Written from MATLAB's documented
% gpuArray behavior; runs in numbl on the CPU reference device.

G = gpuArray([1 2 3]);
assert(strcmp(class(G), 'gpuArray') && isgpuarray(G) && ~isgpuarray([1 2 3]));
assert(strcmp(underlyingType(G), 'double') && strcmp(underlyingType(1), 'double'));
assert(isUnderlyingType(G, 'double') && isUnderlyingType(G, 'float') && ~isUnderlyingType(G, 'single'));
assert(isa(G, 'gpuArray') && ~isa(G, 'double') && isa(G, 'numeric') && isa(G, 'float'));
assert(isnumeric(G) && isfloat(G) && isreal(G) && ~islogical(G));
assert(isequal(size(G), [1 3]) && numel(G) == 3 && ndims(G) == 2 && length(G) == 3);
assert(~isempty(G) && isvector(G) && isrow(G) && ~iscolumn(G));
assert(strcmp(class(size(G)), 'double') && strcmp(class(numel(G)), 'double'));
assert(gpuArray(G) == G);

% gather
X = gather(G);
assert(strcmp(class(X), 'double') && isequal(X, [1 2 3]));
assert(isequal(gather([4 5]), [4 5]) && strcmp(class(gather(2)), 'double'));
[a, b] = gather(G, 7);
assert(isequal(a, [1 2 3]) && b == 7);
X(1) = 100;
assert(isequal(gather(G), [1 2 3]));

% Results stay on the device, in the underlying class
H = G + 1;
assert(isgpuarray(H) && isequal(gather(H), [2 3 4]));
assert(isgpuarray(1 + G) && isgpuarray(G .* G) && isgpuarray(-G) && isgpuarray(G'));
assert(isgpuarray(sin(G)) && strcmp(underlyingType(sin(G)), 'double'));
assert(isgpuarray(sum(G)) && gather(sum(G)) == 6);
assert(isgpuarray(G > 1) && strcmp(underlyingType(G > 1), 'logical'));
assert(isgpuarray(G * G') && gather(G * G') == 14);
[m, i] = max(G);
assert(isgpuarray(m) && isgpuarray(i) && gather(m) == 3 && gather(i) == 3);
assert(isgpuarray(reshape(G, 3, 1)) && isequal(size(reshape(G, 3, 1)), [3 1]));
assert(isequal(G, [1 2 3]) && islogical(isequal(G, [1 2 3])));

% single on the device
S = gpuArray(single([1 2]));
assert(strcmp(underlyingType(S), 'single'));
assert(strcmp(underlyingType(S * 2.5), 'single') && strcmp(underlyingType(S + G(1:2)), 'single'));
assert(double(gather(S / 3)) * [1; 0] == double(single(1/3)));
assert(strcmp(underlyingType(single(G)), 'single') && strcmp(underlyingType(double(S)), 'double'));
assert(strcmp(class(gather(S)), 'single'));

% Indexing and assignment
x = G(2);
assert(isgpuarray(x) && gather(x) == 2);
assert(isgpuarray(G(end)) && gather(G(end)) == 3 && isequal(gather(G([1 3])), [1 3]));
G2 = G;
G2(2) = 7;
assert(isgpuarray(G2) && isequal(gather(G2), [1 7 3]) && isequal(gather(G), [1 2 3]));
G2(5) = 1;
assert(isequal(gather(G2), [1 7 3 0 1]));
G2(1) = [];
assert(isequal(gather(G2), [7 3 0 1]));
A = zeros(1, 3);
A(2) = gpuArray(5);
assert(isgpuarray(A) && isequal(gather(A), [0 5 0]));

% Concatenation, colon, loops, conditions
C = [G, 4];
assert(isgpuarray(C) && isequal(gather(C), [1 2 3 4]));
assert(isgpuarray([G; G]) && isequal(size([G; G]), [2 3]));
R = gpuArray(1):3;
assert(isgpuarray(R) && isequal(gather(R), [1 2 3]));
n = 0;
for col = gpuArray([1 2; 3 4])
  assert(isgpuarray(col) && isequal(size(col), [2 1]));
  n = n + 1;
end
assert(n == 2);
if G(1) == 1
  ok = true;
else
  ok = false;
end
assert(ok);

% Creation on the device
Z = zeros(2, 3, 'gpuArray');
assert(isgpuarray(Z) && isequal(size(Z), [2 3]) && strcmp(underlyingType(Z), 'double'));
assert(strcmp(underlyingType(ones(2, 'single', 'gpuArray')), 'single'));
assert(isgpuarray(eye(2, 'gpuArray')) && isgpuarray(rand(2, 'gpuArray')));
assert(isgpuarray(randn(1, 2, 'gpuArray')) && isgpuarray(randi(5, 2, 'gpuArray')));
assert(strcmp(underlyingType(true(2, 'gpuArray')), 'logical'));
L = zeros(2, 'like', S);
assert(isgpuarray(L) && strcmp(underlyingType(L), 'single'));
assert(isgpuarray(cast([1 2], 'like', S)) && strcmp(underlyingType(cast([1 2], 'like', S)), 'single'));

% Containers keep gpuArrays
st = struct('a', G);
assert(isgpuarray(st.a));
c = {G};
assert(isgpuarray(c{1}));
f = @(v) v * 2;
assert(isgpuarray(feval(f, G)) && isgpuarray(arrayfun(@(v) v + 1, G)));

% Devices
d = gpuDevice;
assert(ischar(d.Name) && islogical(d.SupportsDouble) && d.Index >= 1);
assert(gpuDeviceCount >= 1 && canUseGPU);
K = gpuArray(1);
assert(existsOnGPU(K));
reset(d);
assert(~existsOnGPU(K));
try
  K + 1;
  error('expected an error');
catch e
  assert(strcmp(e.message, 'The data no longer exists on the device.'));
end

disp('SUCCESS')
