classdef VertCat_
% Helper class for testing that [a; b] calls vertcat on class instances.
  properties
    data
  end
  methods
    function obj = VertCat_(val)
      if nargin > 0
        obj.data = val;
      else
        obj.data = [];
      end
    end
    function result = vertcat(varargin)
      % Concatenate the .data fields vertically
      allData = [];
      for k = 1:nargin
        allData = [allData; varargin{k}.data];
      end
      result = VertCat_(allData);
    end
  end
end
