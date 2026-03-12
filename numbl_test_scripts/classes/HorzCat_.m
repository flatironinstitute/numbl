classdef HorzCat_
% Helper class for testing that [a, b] calls horzcat on class instances.
  properties
    data
  end
  methods
    function obj = HorzCat_(val)
      if nargin > 0
        obj.data = val;
      else
        obj.data = [];
      end
    end
    function result = horzcat(varargin)
      % Concatenate the .data fields horizontally
      allData = [];
      for k = 1:nargin
        allData = [allData, varargin{k}.data];
      end
      result = HorzCat_(allData);
    end
  end
end
