classdef MiBaseA_ < handle
   properties
      a = 1
   end
   methods
      function r = fromA(obj)
         r = obj.a * 10;
      end
   end
end
