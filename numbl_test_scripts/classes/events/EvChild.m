classdef EvChild < EvSource
    % Fixture for the events tests: inherited events, a protected-notify
    % event, an observable property with a set method, a delete method.
    events
        ChildEvt
    end
    properties (SetObservable)
        Clamped = 0
    end
    properties
        Log = []
    end
    methods
        function set.Clamped(obj, v)
            if ~isempty(obj.Log)
                obj.Log.add('set.Clamped');
            end
            obj.Clamped = min(v, 10);
        end
        function fireProt(obj)
            notify(obj, 'ProtBoth');
        end
        function lh = listenProt(obj, cb)
            lh = addlistener(obj, 'ProtBoth', cb);
        end
        function delete(obj)
            if ~isempty(obj.Log)
                obj.Log.add('delete method');
            end
        end
    end
end
