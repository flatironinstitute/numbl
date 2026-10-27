classdef MiChild_ < MiBaseA_ & MiBaseB_
   properties
      c = 3
   end
   methods
      function r = total(obj)
         r = obj.a + obj.b + obj.c;
      end
   end
end
