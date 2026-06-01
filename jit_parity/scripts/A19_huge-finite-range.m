% DIAGNOSIS: (same family as 02) The interpreter eagerly materializes a for-loop
%   range as a full tensor (makeRangeTensor allocates n doubles), so a huge but
%   FINITE count fails at allocation even when the body breaks immediately. The
%   JS-JIT iterates the range lazily, so the early `break` makes opt1 succeed.
% --opt 0 output: RuntimeError: Array buffer allocation failed
% --opt 1 output: 200
hit=0; for rep=1:200; for k=1:1e15; hit=hit+1; break; end; end; disp(hit)
