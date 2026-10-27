classdef MiBaseB_ < matlab.mixin.Copyable
   properties
      b = 2
   end
   methods
      function r = fromB(obj)
         r = obj.b * 100;
      end
      function setB(obj, v)
         obj.b = v;
      end
   end
end
