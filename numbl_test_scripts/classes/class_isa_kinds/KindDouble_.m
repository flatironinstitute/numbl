classdef KindDouble_ < double
    methods
        function obj = KindDouble_(d)
            if nargin < 1
                d = 0;
            end
            obj = obj@double(d);
        end
    end
end
