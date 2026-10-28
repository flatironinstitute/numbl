% DIAGNOSIS: A loop that draws rand() and then grow-bails is re-run from the
% start by the interpreter, but the JIT attempt's draws had already advanced
% the shared PRNG, so the re-run drew different numbers than --opt 0. Fixed by
% snapshotting the PRNG before a JIT attempt and restoring it on bail. At
% --opt 2 the C-JIT also crashed on rand (no emitC) instead of declining to
% JS-JIT.
%
% --opt 0 stdout:
% 1.933359608752653
% 0.473531235242262
%
% --opt 1 stdout:
% 2.753775770543143
% 0.224522636272013
rng(7);
v = zeros(1,3);
s = 0;
for k=1:5
  s = s + rand();
  v(k) = k;   % grows past size 3 -> runtime grow-bail
end
fprintf('%.15f\n', s);
fprintf('%.15f\n', rand());
