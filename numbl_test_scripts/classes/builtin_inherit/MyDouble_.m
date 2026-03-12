classdef MyDouble_ < double
    properties
        label
    end
    methods
        function obj = MyDouble_(data, lbl)
            if nargin < 1
                data = [];
            end
            obj = obj@double(data);
            if nargin >= 2
                obj.label = lbl;
            else
                obj.label = '';
            end
        end
        function d = getData(obj)
            d = double(obj);
        end
    end
end
