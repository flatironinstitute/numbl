classdef ParenAsgnBypassSubsasgnClass_
% Class to test that obj(k) = same_class_val inside a class method bypasses
% the overloaded subsasgn (mirrors the MATLAB classdef rule: inside a class
% method, direct subscript assignment uses built-in array mechanics).
    properties
        val
    end
    methods
        function obj = ParenAsgnBypassSubsasgnClass_(v)
            if nargin == 0
                obj.val = 0;
            else
                obj.val = v;
            end
        end

        function obj = subsasgn(obj, S, rhs)
            if strcmp(S(1).type, '()')
                error('subsasgn called with () — should not happen for same-class assignment inside class method');
            end
            obj = builtin('subsasgn', obj, S, rhs);
        end

        function obj = replaceFirst(obj, other)
            % Inside a class method, obj(1) = other should NOT call subsasgn.
            obj(1) = other;
        end

        function F = loopReplace(F)
            % Reassign F via a function call (makes type Unknown in IR),
            % then F(k) = val inside loop — same pattern as chebfun/restrict.m.
            F = ParenAsgnBypassSubsasgnClass_.identity(F);
            for k = 1:numel(F)
                F(k) = ParenAsgnBypassSubsasgnClass_.transform(F(k));
            end
        end
    end

    methods (Static)
        function out = identity(obj)
            out = obj;
        end
        function obj = transform(obj)
            obj.val = obj.val * 2;
        end
    end
end
