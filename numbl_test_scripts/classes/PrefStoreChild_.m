classdef PrefStoreChild_ < PrefStoreParent_
% Child class that inherits prefList from parent.
% Has subsref/subsasgn that access obj.prefList internally.
% Mimics chebfunpref < chebpref pattern where prefList is in parent.
    properties
    end

    methods
        function obj = PrefStoreChild_(varargin)
            if nargin == 1 && isa(varargin{1}, 'PrefStoreChild_')
                obj = varargin{1};
                return
            elseif nargin == 1 && isstruct(varargin{1})
                inPrefList = varargin{1};
            else
                inPrefList = struct();
            end
            obj.prefList = struct();
            obj.prefList.alpha = 10;
            obj.prefList.techPrefs = struct();
            obj.prefList.techPrefs.x = 1;
            % Copy fields from input struct
            for field = fieldnames(inPrefList).'
                if isfield(obj.prefList, field{1})
                    obj.prefList.(field{1}) = inPrefList.(field{1});
                else
                    obj.prefList.techPrefs.(field{1}) = inPrefList.(field{1});
                end
            end
        end

        function out = subsref(obj, S)
            switch S(1).type
                case '.'
                    if isfield(obj.prefList, S(1).subs)
                        out = obj.prefList.(S(1).subs);
                        if numel(S) > 1
                            out = builtin('subsref', out, S(2:end));
                        end
                    else
                        error('No field %s', S(1).subs);
                    end
                otherwise
                    error('Unsupported subscript type');
            end
        end

        function obj = subsasgn(obj, S, val)
            switch S(1).type
                case '.'
                    if isfield(obj.prefList, S(1).subs)
                        obj.prefList = builtin('subsasgn', obj.prefList, S, val);
                    else
                        obj.prefList.techPrefs = builtin('subsasgn', ...
                            obj.prefList.techPrefs, S, val);
                    end
                otherwise
                    error('Unsupported subscript type');
            end
        end
    end
end
