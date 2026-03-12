% A class with a 'compute' method. Its presence causes calls to compute()
% with an Unknown-typed first argument to go through runtime dispatch via
% $primaryFunctions, which is where the local-function shadowing bug manifests.
classdef MyCompute
  methods
    function result = compute(self, x)
      result = x + 1;
    end
  end
end
