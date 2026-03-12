classdef SimpleCtorTest_
    properties
        Value
        Initialized
    end
    methods
        function obj = SimpleCtorTest_(v)
            obj.Value = v * 2;
            obj.Initialized = 1;
        end
    end
end
