% The gpuArray functions that work on host values, without a GPU device.
% Verified against MATLAB R2026b (Parallel Computing Toolbox, no GPU).

assert(strcmp(underlyingType(1), 'double') && strcmp(underlyingType(single(1)), 'single'));
assert(strcmp(underlyingType(true), 'logical') && strcmp(underlyingType('a'), 'char'));
assert(isUnderlyingType(single(1), 'float') && isUnderlyingType(single(1), 'single'));
assert(isUnderlyingType(1, 'numeric') && ~isUnderlyingType(1, 'single'));
assert(gather(5) == 5 && strcmp(class(gather(single(1))), 'single'));
[a, b] = gather(1, 2);
assert(a == 1 && b == 2);
assert(~isgpuarray(1) && ~isgpuarray(single([1 2])));
try
  existsOnGPU(1);
  error('expected an error');
catch e
  assert(strcmp(e.message, 'Undefined function ''existsOnGPU'' for input arguments of type ''double''.'));
end

disp('SUCCESS')
