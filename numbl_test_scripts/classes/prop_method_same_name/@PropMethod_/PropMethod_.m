classdef PropMethod_
    properties
        value
        domain
    end
    methods
        function obj = PropMethod_(v)
            obj.value = v;
            obj.domain = [1 2 3 4 5];
        end
    end
end
