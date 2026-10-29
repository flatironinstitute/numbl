% A GPU device without double precision: creating a double gpuArray is an
% error, single works, and results of operations are not restricted.
% numbl's device 2 is the CPU reference device without double precision.

if exist('isnumbl', 'builtin') ~= 5
  disp('SUCCESS')
  return
end

% A device without double precision (a WebGPU device, or numbl's CPU
% reference device that acts as one) and one with.
single_dev = 0;
double_dev = 0;
for k = 1:gpuDeviceCount
  dk = gpuDevice(k);
  if dk.SupportsDouble && double_dev == 0
    double_dev = k;
  elseif ~dk.SupportsDouble && single_dev == 0
    single_dev = k;
  end
end
d = gpuDevice(single_dev);
assert(~d.SupportsDouble && d.Index == single_dev && d.DeviceSelected);
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
gpuDevice(double_dev);
assert(~existsOnGPU(S));
d = gpuDevice;
assert(d.SupportsDouble && d.Index == double_dev);
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
