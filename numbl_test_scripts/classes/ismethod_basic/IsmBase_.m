classdef IsmBase_ < handle
   properties
      v = 0
   end
   methods
      function r = basepub(obj)
         r = 1;
      end
   end
   methods (Access = private)
      function r = basepriv(obj)
         r = 2;
      end
   end
end
