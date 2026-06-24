classdef ConvLeaf_ < ConvBase_
  % Inherits a method named ConvLeaf_ from ConvBase_. The constructor copies
  % when handed an existing ConvLeaf_.
  methods
    function obj = ConvLeaf_(varargin)
      obj.data = 0;
      if length(varargin) == 1
        a = varargin{1};
        if isa(a, 'ConvLeaf_')
          obj.data = a.data;
        end
      end
    end
  end
end
