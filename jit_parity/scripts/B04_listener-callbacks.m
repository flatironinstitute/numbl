% Event listeners around JIT-able code. notify/addlistener are interpreter
% only: a loop that triggers events must decline (running it compiled and
% bailing would re-run the callbacks' output), while a numeric function
% called from a callback may compile. An error in such a callback (raised
% from compiled code at opt 1/2) becomes the listener warning in every mode,
% and the loop continues.
s = B04Source();
lh = addlistener(s, 'Tick', @(src, ~) fprintf('tick %d -> %g\n', src.N, b04_work(src.N)));
for k = 1:4
    s.N = k;
    notify(s, 'Tick');
end
total = 0;
for k = 1:1000
    total = total + b04_work(mod(k, 3) + 1);
end
fprintf('total %g\n', total);
p = addlistener(s, 'N', 'PostSet', @(src, evt) fprintf('N set to %d\n', evt.AffectedObject.N));
for k = 1:3
    s.N = k * 10;
end
delete(lh);
delete(p);
lh2 = addlistener(s, 'Tick', @(src, ~) b04_work(-src.N));
s.N = 2;
notify(s, 'Tick');
disp('after error');

function y = b04_work(n)
if n < 0
    error('b04:neg', 'negative input %d', n);
end
y = 0;
for i = 1:n
    y = y + i * 0.5;
end
end

classdef B04Source < handle
    events
        Tick
    end
    properties (SetObservable)
        N = 0
    end
end
