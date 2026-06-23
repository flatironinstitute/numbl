classdef SuperParent_ < SuperGrandparent_
  properties
    B
  end
  methods
    function obj = SuperParent_(a, b)
      if nargin == 0
        args = {};
      else
        args = {a};
      end
      obj = obj@SuperGrandparent_(args{:});
      if nargin > 1
        obj.B = b;
      end
    end
  end
end
