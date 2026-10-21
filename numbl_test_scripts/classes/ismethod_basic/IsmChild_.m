classdef IsmChild_ < IsmBase_
   properties
      w = 1
   end
   methods
      function obj = IsmChild_()
      end
      function r = pub(obj)
         r = 1;
      end
      function set.w(obj, val)
         obj.w = val;
      end
   end
   methods (Access = protected)
      function r = prot(obj)
         r = 2;
      end
   end
   methods (Static)
      function r = stat()
         r = 3;
      end
   end
end
