classdef Num2cellObj_
   properties
      v = 0
   end
   methods
      function obj = Num2cellObj_(v)
         if nargin > 0
            obj.v = v;
         end
      end
   end
end
