% A GPU device without double precision: creating a double gpuArray is an
% error, single works, and results of operations are not restricted.
% numbl's device 2 is the CPU reference device without double precision.

if exist('isnumbl', 'builtin') ~= 5
  disp('SUCCESS')
  return
end

d = gpuDevice(2);
assert(~d.SupportsDouble && d.Index == 2 && d.DeviceSelected);
msg = 'The selected GPU device does not support double precision. Convert the data to single first, for example gpuArray(single(X)).';
expectError(@() gpuArray([1 2]), msg);
expectError(@() zeros(2, 'gpuArray'), msg);
expectError(@() rand(2, 'gpuArray'), msg);
S = gpuArray(single([1 2 3]));
expectError(@() double(S), msg);
expectError(@() cast(S, 'double'), msg);

% single and logical arrays work, and stay single
assert(strcmp(underlyingType(S), 'single'));
assert(strcmp(underlyingType(S + 1), 'single') && strcmp(underlyingType(sqrt(S)), 'single'));
assert(strcmp(underlyingType(zeros(2, 'single', 'gpuArray')), 'single'));
assert(strcmp(underlyingType(gpuArray(true)), 'logical'));
% an index output is double
[~, i] = max(S);
assert(isgpuarray(i) && strcmp(underlyingType(i), 'double') && gather(i) == 3);

% selecting a device resets it
gpuDevice(1);
assert(~existsOnGPU(S));
d = gpuDevice;
assert(d.SupportsDouble && d.Index == 1);
assert(isgpuarray(gpuArray([1 2])));

disp('SUCCESS')

function expectError(f, msg)
  try
    f();
  catch e
    assert(strcmp(e.message, msg), e.message);
    return
  end
  error('expected an error');
end
