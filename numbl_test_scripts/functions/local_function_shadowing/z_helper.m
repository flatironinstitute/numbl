% This file has a primary function and a local function also named 'compute'.
% The local 'compute' should NOT be callable from outside this file.
% If the local function shadows compute.m's primary function, compute(x)
% would return x*999 instead of x*10.
function result = z_helper(x)
  result = x;
end

function result = compute(x)
  % Local function - must not be reachable from other files
  result = x * 999;
end
