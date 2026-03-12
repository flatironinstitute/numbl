classdef Processor_
    properties
        Factor
    end
    methods
        function obj = Processor_(f)
            obj.Factor = f;
        end
        function r = process(op, values, obj)
            % Method where 'obj' (the class instance) is the 3rd argument
            r = op(values) * obj.Factor;
        end
    end
end
