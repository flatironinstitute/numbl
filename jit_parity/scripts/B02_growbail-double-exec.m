% DIAGNOSIS: A runtime "grow bail" (indexed-store array growth) in the
% whole-scope JIT fires AFTER earlier side effects already executed. The
% interpreter then re-runs the entire script from scratch, so disp(111)
% prints twice. Any runtime bail after a side effect duplicates output.
% (stdout only; warnings go to stderr.)
%
% --opt 0 stdout:
% 111
% 15
%
% --opt 1 stdout:
% 111
% 111
% 15
disp(111)
v = zeros(1,3);
for k=1:5
  v(k) = k;   % grows past size 3 -> runtime grow-bail
end
disp(sum(v));
