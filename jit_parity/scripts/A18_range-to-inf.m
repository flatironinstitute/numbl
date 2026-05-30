% DIAGNOSIS: `for k=1:Inf` — interp computes count via makeRangeTensor (no
%   finiteness guard) and tries to allocate an Inf-length array -> RuntimeError;
%   JS-JIT's mtoc2_loop_count returns 0 for a non-finite count, so the loop body
%   never runs and opt1 succeeds.
% --opt 0 output: RuntimeError: Invalid typed array length: Infinity
% --opt 1 output: 0
hit=0; for rep=1:200; for k=1:Inf; hit=hit+1; break; end; end; disp(hit)
