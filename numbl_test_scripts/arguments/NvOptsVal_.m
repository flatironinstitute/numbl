classdef NvOptsVal_
  methods (Static)
    function r = scaled(x, opts)
      arguments
        x (1,1) double
        opts.factor (1,1) double {mustBePositive} = 1
      end
      r = x * opts.factor;
    end
  end
end
