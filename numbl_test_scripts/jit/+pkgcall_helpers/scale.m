function y = scale(x, k)
% Simple package fn that returns a scalar — used to exercise the path
% where the callee soft-bails to UserDispatchCall (probe) but the outer
% loop still lowers. (Body is trivial; both numbl and a hypothetical
% interpreter-only path return the same scalar.)
y = sum(x(:)) * k;
end
