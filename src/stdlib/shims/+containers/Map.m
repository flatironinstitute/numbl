classdef Map < handle
    properties
        Count
        KeyType
        ValueType
    end
    properties (Access = private)
        data_
    end
    methods
        function obj = Map(varargin)
            obj.Count = 0;
            obj.KeyType = 'char';
            obj.ValueType = 'any';
            obj.data_ = dictionary;

            if nargin == 0
                return;
            end

            % containers.Map('KeyType',kType,'ValueType',vType) or reverse order
            if nargin == 4 && ischar(varargin{1})
                for i = 1:2:nargin
                    key = varargin{i};
                    val = varargin{i+1};
                    if strcmpi(key, 'KeyType')
                        obj.KeyType = val;
                    elseif strcmpi(key, 'ValueType')
                        obj.ValueType = val;
                    end
                end
                if strcmp(obj.ValueType, 'any')
                    obj.data_ = configureDictionary("string", "cell");
                else
                    obj.data_ = configureDictionary("string", mapValueType(obj.ValueType));
                end
                return;
            end

            % containers.Map(keySet, valueSet) or containers.Map(keySet, valueSet, 'UniformValues', false)
            keySet = varargin{1};
            valueSet = varargin{2};
            isUniform = true;
            if nargin >= 4
                for i = 3:2:nargin
                    if strcmpi(varargin{i}, 'UniformValues')
                        isUniform = varargin{i+1};
                    end
                end
            end

            % Determine key type
            if iscell(keySet)
                obj.KeyType = 'char';
            elseif ischar(keySet)
                obj.KeyType = 'char';
                keySet = {keySet};
            else
                obj.KeyType = 'double';
            end

            % Determine value type
            if ~isUniform
                obj.ValueType = 'any';
            elseif iscell(valueSet)
                % Cell of chars → 'char', otherwise 'any'
                allChar = true;
                for i = 1:length(valueSet)
                    if ~ischar(valueSet{i})
                        allChar = false;
                        break;
                    end
                end
                if allChar
                    obj.ValueType = 'char';
                else
                    obj.ValueType = 'any';
                end
            elseif ischar(valueSet)
                obj.ValueType = 'char';
            else
                obj.ValueType = 'double';
            end

            % Build dictionary
            if strcmp(obj.ValueType, 'any')
                obj.data_ = configureDictionary("string", "cell");
            else
                obj.data_ = configureDictionary("string", mapValueType(obj.ValueType));
            end

            % Insert entries
            if iscell(keySet)
                for i = 1:length(keySet)
                    k = makeKey(keySet{i}, obj.KeyType);
                    if strcmp(obj.ValueType, 'any')
                        if iscell(valueSet)
                            obj.data_(k) = valueSet(i);
                        else
                            obj.data_(k) = {valueSet(i)};
                        end
                    else
                        if iscell(valueSet)
                            obj.data_(k) = valueSet{i};
                        else
                            obj.data_(k) = valueSet(i);
                        end
                    end
                end
            else
                % Numeric keys
                for i = 1:length(keySet)
                    k = makeKey(keySet(i), obj.KeyType);
                    if strcmp(obj.ValueType, 'any')
                        if iscell(valueSet)
                            obj.data_(k) = valueSet(i);
                        else
                            obj.data_(k) = {valueSet(i)};
                        end
                    else
                        if iscell(valueSet)
                            obj.data_(k) = valueSet{i};
                        else
                            obj.data_(k) = valueSet(i);
                        end
                    end
                end
            end

            obj.Count = numEntries(obj.data_);
        end

        function result = subsref(obj, S)
            if strcmp(S(1).type, '()')
                k = makeKey(S(1).subs{1}, obj.KeyType);
                val = obj.data_(k);
                if strcmp(obj.ValueType, 'any')
                    result = val{1};
                else
                    result = val;
                end
            elseif strcmp(S(1).type, '.')
                result = builtin('subsref', obj, S);
            elseif strcmp(S(1).type, '{}')
                result = builtin('subsref', obj, S);
            end
        end

        function obj = subsasgn(obj, S, val)
            if strcmp(S(1).type, '()')
                k = makeKey(S(1).subs{1}, obj.KeyType);
                if strcmp(obj.ValueType, 'any')
                    obj.data_(k) = {val};
                else
                    obj.data_(k) = val;
                end
                obj.Count = numEntries(obj.data_);
            elseif strcmp(S(1).type, '.')
                obj = builtin('subsasgn', obj, S, val);
            end
        end

        function result = isKey(obj, key)
            k = makeKey(key, obj.KeyType);
            result = isKey(obj.data_, k);
        end

        function result = keys(obj)
            if obj.Count == 0
                result = {};
                return;
            end
            k = keys(obj.data_);
            if strcmp(obj.KeyType, 'char')
                % Return cell array of char vectors
                if iscell(k)
                    result = k;
                else
                    result = {k};
                end
            else
                % Numeric keys: parse back from string
                if iscell(k)
                    result = cell(size(k));
                    for i = 1:length(k)
                        result{i} = str2double(k{i});
                    end
                else
                    result = {str2double(k)};
                end
            end
        end

        function result = values(obj)
            if obj.Count == 0
                result = {};
                return;
            end
            v = values(obj.data_);
            if strcmp(obj.ValueType, 'any')
                result = v;
            else
                if iscell(v)
                    result = v;
                else
                    % Wrap scalars in cells
                    result = cell(1, obj.Count);
                    for i = 1:obj.Count
                        result{i} = v(i);
                    end
                end
            end
        end

        function result = length(obj)
            result = obj.Count;
        end

        function [m, n] = size(obj)
            m = obj.Count;
            n = 1;
        end

        function obj = remove(obj, keyOrKeys)
            if iscell(keyOrKeys)
                for i = 1:length(keyOrKeys)
                    k = makeKey(keyOrKeys{i}, obj.KeyType);
                    obj.data_(k) = [];
                end
            else
                k = makeKey(keyOrKeys, obj.KeyType);
                obj.data_(k) = [];
            end
            obj.Count = numEntries(obj.data_);
        end
    end
end

function k = makeKey(raw, keyType)
    if strcmp(keyType, 'char')
        if ischar(raw)
            k = raw;
        else
            k = char(raw);
        end
    else
        k = num2str(raw);
    end
end

function vt = mapValueType(vType)
    if strcmp(vType, 'double') || strcmp(vType, 'single')
        vt = "double";
    elseif strcmp(vType, 'char')
        vt = "string";
    elseif strcmp(vType, 'logical')
        vt = "double";
    else
        vt = "cell";
    end
end
