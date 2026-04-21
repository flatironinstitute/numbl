% C-JIT parity gap #18: UserCall with complex tensor args and/or complex
% tensor return. Builds on parity13 (real-tensor UserCall) and parity15/16
% (complex-tensor locals) by wiring the paired imag buffer through the
% callee ABI — both on the arg side (caller's v_z_data_im → callee's
% tensorDataIm slot) and on the return side (callee's dynOutBufIm
% transfers ownership back into the caller's complex destination).
%
% Expected disp output (should match across all runs):
%   numbl --opt 1 run <this>                             -> values below
%   numbl --opt 2 run <this>                             -> values below
%   numbl --opt 2 --check-c-jit-parity run <this>        -> values below
%   matlab -batch parity18_user_call_complex_tensor      -> values below

function parity18_user_call_complex_tensor()
    a = [1.0, 2.0, 3.0];

    % 1) Complex tensor arg + real scalar return. Exercises the arg side
    %    of the paired-ABI (caller passes tensorDataIm; callee's abs+sum
    %    drops the imag in the aggregation).
    s = 0.0;
    for k = 1:5
        z = a + a * 1i;      % [1+i, 2+2i, 3+3i]
        s = s + abs_sum(z);  % sqrt(2) * (1+2+3) = 6*sqrt(2) ≈ 8.4853
    end
    disp(s / 5)   % 6*sqrt(2)

    % 2) Complex tensor arg + complex tensor return. Exercises both arg
    %    and return sides: callee's dynamic output transfers two buffers
    %    (re + im) back via paired `double **` out-slots.
    t_re = 0.0; t_im = 0.0;
    for k = 1:5
        z = a + a * 1i;              % [1+i, 2+2i, 3+3i]
        w = scale_complex(z, 2.0);    % [2+2i, 4+4i, 6+6i]
        t = sum(w);                   % 12 + 12i
        t_re = t_re + real(t);
        t_im = t_im + imag(t);
    end
    disp(t_re / 5)   % 12
    disp(t_im / 5)   % 12

    disp('SUCCESS')
end

function s = abs_sum(z)
    s = sum(abs(z));
end

function w = scale_complex(z, s)
    w = z * s;
end
