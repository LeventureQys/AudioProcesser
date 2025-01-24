clear all; clc;close all
a = 0:1:1023;
format longG

for i = 1:1:10
    a_con = [];
    for j = 1:1:2^(i-1)
        len = length(a)/(2^(i-1));
        a_init = a(1+(j-1)*len: j*len);
        a_res1 = a_init(1:2:end);
        a_res2 = a_init(2:2:end);
        a_con = [a_con a_res1 a_res2];
    end
    a = a_con;
end

N_FFT = 1024;
% digits(6); 

w = 2 * (0:1:N_FFT-1) * pi / N_FFT;
cos_table = sin(w);
raw = 16;
for i=1:N_FFT/raw
    idx = 1 + (i-1)*raw: i*raw;
    str = join(string(cos_table(idx)),', ');
    fprintf("%s\n", str);
end