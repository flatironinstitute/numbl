classdef QrObj_
    properties
        val
    end
    methods
        function obj = QrObj_(v)
            obj.val = v;
        end
        function result = horzcat(varargin)
            s = 0;
            for i = 1:nargin
                s = s + varargin{i}.val;
            end
            result = QrObj_(s);
        end
        function result = qr(obj)
            result = obj.val * 10;
        end
    end
end
