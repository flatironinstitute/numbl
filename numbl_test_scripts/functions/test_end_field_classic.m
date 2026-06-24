function test_end_field_classic()
% Regression: 'end' used as a struct field name (e.g. s.end) must not be
% miscounted as a block closer in classic-style (non-end-terminated)
% function files. This previously caused a spurious "expected 'end'" parse
% error on jsonlab's jsonpath.m.
%
% This is a classic-style function file (no function is closed with 'end').
% Run it by invoking the function: numbl auto-invokes the entry function;
% in MATLAB use `matlab -batch "test_end_field_classic"`.
r = compute();
assert(r == 8, 'end-field arithmetic');
assert(strcmp(label(), 'done'), 'end-field in a second classic function');
disp('SUCCESS')

function out = compute()
s.start = 3;
s.end = 5;
s.end = s.end + 0;
out = s.start + s.end;

function s = label()
m.end = 'done';
s = m.end;
