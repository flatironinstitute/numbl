% TEST: %x / %o of a negative half-way value.
%   %x of -1.5, %x of -2.5, %o of -1.5
%   opt0/1: "1|2|1"   (JS Math.round rounds .5 toward +Inf, then Math.abs:
%                       round(-1.5)=-1 ->1 ; round(-2.5)=-2 ->2)
%   opt2:   "2|3|2"   (C floor(fabs(raw)+0.5): round-half-up on magnitude:
%                       |−1.5|->2 ; |−2.5|->3)
% DIVERGING MODES: opt2 vs opt0/opt1.
% CAUSE: JS rounds the signed value with Math.round (ties toward +Inf) then
%        takes |.|; C rounds the magnitude with floor(fabs+0.5) (ties up).
% JIT ENGAGEMENT: confirmed (--dump-c non-empty).
fprintf('%x|%x|%o\n', -1.5, -2.5, -1.5);
