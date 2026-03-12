classdef BuiltinSubsasgnClass_
% Test class that mirrors the chebfun pattern: overloaded subsasgn whose
% dot-case delegates to builtin('subsasgn', obj, index, val) on the class
% instance itself (not a plain struct). builtin() must bypass the class
% method to avoid infinite recursion.
    properties
        data
        domain
    end

    methods
        function obj = BuiltinSubsasgnClass_()
            obj.data = [];
            obj.domain = [-1, 1];
        end

        function obj = subsasgn(obj, index, val)
            switch index(1).type
                case '.'
                    % Delegate raw property assignment to builtin, bypassing
                    % this overloaded method (mirrors chebfun/subsasgn.m).
                    obj = builtin('subsasgn', obj, index, val);
                otherwise
                    error('Unsupported subscript type: %s', index(1).type);
            end
        end

        function out = subsref(obj, index)
            switch index(1).type
                case '.'
                    % Delegate raw property read to builtin, bypassing this
                    % overloaded method (mirrors chebfun/subsref.m).
                    out = builtin('subsref', obj, index);
                otherwise
                    error('Unsupported subscript type: %s', index(1).type);
            end
        end
    end
end
