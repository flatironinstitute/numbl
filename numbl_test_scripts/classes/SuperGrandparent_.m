classdef SuperGrandparent_
  properties
    A
  end
  methods
    function obj = SuperGrandparent_(a)
      if nargin > 0
        obj.A = a;
      end
    end
  end
end
