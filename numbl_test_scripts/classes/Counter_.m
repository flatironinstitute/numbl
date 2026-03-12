classdef Counter_ < handle
   properties
      Value = 0
   end
   methods
      function increment(obj, amount)
         if nargin < 2
            amount = 1;
         end
         obj.Value = obj.Value + amount;
      end
      function reset(obj)
         obj.Value = 0;
      end
   end
end
