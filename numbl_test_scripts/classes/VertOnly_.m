classdef VertOnly_
% Helper class that supports vertcat but NOT horzcat.
% Used to test that [a; b; c] does not spuriously call horzcat.
  properties
    data
  end
  methods
    function obj = VertOnly_(val)
      if nargin > 0
        obj.data = val;
      else
        obj.data = [];
      end
    end
    function result = vertcat(varargin)
      allData = [];
      for k = 1:nargin
        allData = [allData; varargin{k}.data];
      end
      result = VertOnly_(allData);
    end
    function result = horzcat(varargin)
      error('VertOnly_:horzcat', 'horzcat is not supported');
    end
  end
end
