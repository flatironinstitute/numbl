classdef EvSource < handle
    % Fixture for the events tests: events with every access attribute and
    % observable properties.
    events
        Evt1
        Evt2
    end
    events (NotifyAccess = private)
        PrivNotify
    end
    events (ListenAccess = private)
        PrivListen
    end
    events (ListenAccess = protected, NotifyAccess = protected)
        ProtBoth
    end
    events (Hidden)
        HidEvt
    end
    properties (SetObservable, GetObservable)
        Val = 1
    end
    properties (SetObservable)
        Other = 0
    end
    properties
        Plain = 5
    end
    properties (Access = private)
        Secret = 0
    end
    methods
        function fire(obj, name)
            notify(obj, name);
        end
        function fireWith(obj, name, data)
            notify(obj, name, data);
        end
        function setOther(obj, v)
            obj.Other = v;
        end
        function lh = listenPriv(obj, cb)
            lh = addlistener(obj, 'PrivListen', cb);
        end
    end
end
