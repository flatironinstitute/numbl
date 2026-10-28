% TEST: warnings from compiled code respect the warning state.
%
% The JITs printed every warning: warning('off', id) was ignored, and
% warning('off', id) / warning('on') inside a compiled unit printed
% "Warning: off" / "Warning: on". The runner drops lines starting with
% "warning:", so each message carries a second line that stays visible.
warning('off', 'gp:quiet');
s = 0;
for k = 1:3
  s = s + k;
  warning('gp:quiet', 'x\nquiet %d', k);
  warning('gp:loud', 'x\nloud %d', k);
  warning('x\nbare %d', k);
end
r = in_function(2);
t = toggles(2);
warning('off', 'all');
r = in_function(2);
warning('on', 'all');
r = in_function(1);
fprintf('%d %d %d\n', s, r, t);

function s = in_function(n)
  s = 0;
  for k = 1:n
    s = s + k;
    warning('gp:quiet', 'x\nfn quiet %d', k);
    warning('gp:loud', 'x\nfn loud %d', k);
  end
end

function s = toggles(n)
  s = 0;
  for k = 1:n
    s = s + k;
    warning('gp:toggled', 'x\ntoggled %d', k);
    warning('off', 'gp:toggled');
  end
end
