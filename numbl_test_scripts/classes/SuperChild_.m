classdef SuperChild_ < SuperParent_
  properties
    C
  end
  methods
    function obj = SuperChild_(a, b, c)
      if nargin == 0
        args = {};
      else
        args = {a, b};
      end
      obj = obj@SuperParent_(args{:});
      if nargin > 2
        obj.C = c;
      end
    end
  end
end
