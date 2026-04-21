% C-JIT parity gap #16: complex tensor unary + conj / real / imag.
%
% Phase 2 adds inline emission for:
%   -z       → complex scalar MUL by -1
%   conj(z)  → copy re, negate im
%   real(z)  → memcpy re into a fresh real tensor
%   imag(z)  → memcpy im into a fresh real tensor (or zero if NULL imag)
%
% Expected disp output (should match across all runs):
%   numbl --opt 1 run <this>                         -> values below
%   numbl --opt 2 run <this>                         -> values below
%   numbl --opt 2 --check-c-jit-parity run <this>    -> values below
%   matlab -batch parity16_complex_tensor_unary      -> values below

function parity16_complex_tensor_unary()
    a = [1.0, 2.0, 3.0];

    % 1) Unary minus on a complex tensor — kernel-backed.
    s_re = 0.0; s_im = 0.0;
    for k = 1:5
        z = a + a * 1i;          % [1+i, 2+2i, 3+3i]
        w = -z;                  % [-1-i, -2-2i, -3-3i]
        wr = real(w);
        wi = imag(w);
        s_re = s_re + wr(2);     % -2
        s_im = s_im + wi(2);     % -2
    end
    disp(s_re / 5)   % -2
    disp(s_im / 5)   % -2

    % 2) conj — inline loop.
    c_re = 0.0; c_im = 0.0;
    for k = 1:5
        z = a + a * 1i;
        cz = conj(z);            % [1-i, 2-2i, 3-3i]
        cr = real(cz);
        ci = imag(cz);
        c_re = c_re + cr(3);     % 3
        c_im = c_im + ci(3);     % -3
    end
    disp(c_re / 5)   % 3
    disp(c_im / 5)   % -3

    % 3) real / imag extraction. imag on a real-widened tensor is zero.
    r_re = 0.0; r_im = 0.0;
    for k = 1:5
        z = a + 0 * 1i;          % complex-typed but all imag = 0
        zr = real(z);
        zi = imag(z);
        r_re = r_re + zr(1);     % 1
        r_im = r_im + zi(1);     % 0
    end
    disp(r_re / 5)   % 1
    disp(r_im / 5)   % 0

    disp('SUCCESS')
end
