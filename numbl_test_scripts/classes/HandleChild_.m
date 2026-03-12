classdef HandleChild_ < Counter_
   properties
      Name = 'default'
   end
   methods
      function set_name(obj, n)
         obj.Name = n;
      end
   end
end
