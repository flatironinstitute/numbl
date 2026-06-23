classdef Derived < shp.Base
    properties
        w = 0
    end
    methods
        function obj = Derived(x, y)
            args = {};
            if nargin > 0
                args = {x};
            end
            % Package-qualified superclass constructor call.
            obj = obj@shp.Base(args{:});
            if nargin > 1
                obj.w = y;
            end
        end
        function s = describe(obj)
            % Package-qualified superclass method call.
            base = describe@shp.Base(obj);
            s = sprintf('%s+Derived(%g)', base, obj.w);
        end
    end
end
