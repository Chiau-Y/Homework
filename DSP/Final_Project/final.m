clc;
clear;

fs = 44000;                                           % �ļ��W�v
% %--------------------LPF--------------------%
% fc1 = 2000;                                           % �I���W�v(200Hz)
% num = 2*pi*fc1;
% den = [1 2*pi*fc1];
% H1 = tf(num,den);                                     % �s��������
% Hd1 = c2d(H1,1/fs);
% 
% %--------------------HPF--------------------%
% fc2 = 1000;                                           % �I���W�v(1000Hz)
% num = [1 0];
% den = [1 2*pi*fc2];
% H2 = tf(num,den);
% Hd2 = c2d(H2,1/fs);                                   % �s��������
% 
% Hd2*Hd2*Hd1*Hd1
% 
% %-----------------Bode Plot-----------------%
% % bode(Hd1*Hd1);hold on; 
% % bode(Hd2*Hd2);
% % legend('LPF','HPF');
% % bode(Hd2*Hd2*Hd1*Hd1);
% % bode(Hd1);hold on; 
% % bode(Hd2);
% % legend('LPF','HPF');
% 
%-----------------Sine Wave-----------------%
A = 1.0;                                                   % sin�i���T
f1 = 100;
w = f1*2*pi;                                                % sin�i?�v
ph = 0;                                                    % sin�i����l�ۦ�

t = 0;
x = zeros(1,fs);                                           % ���ͪ�sin�i
for m=1:fs-1
    x(m)=A*sin(w*t+ph);
    t=t+1/fs;
end

y3 = zeros(1,fs-1);        
for i = 5 : fs-1
    y3(i) = 2.109 * y3(i-3) - 3.923 * y3(i-2) + 3.237 * y3(i-1) - 0.4245 * y3(i-4) + 0.06172 * x(i-2) - 0.1234 * x(i-3) + 0.06172 * x(i-4);
end
%                            
% % plot(x(60:200)); hold on;
% % plot(y3(60:200));
% % legend('sin�i','After HPF & LPF');  

%-------------------FFT-------------------%
N = 43999;			    % length of vector (�I��)
freqStep = fs/N;		% freq resolution in spectrum (�W�쪺�W�v���ѪR��)
f = 10*freqStep;		% freq of the sinusoid (�����i���W�v�A��O freqStep ����ƭ�)
time = (0:N-1)/fs;		% time resolution in time-domain (�ɰ쪺�ɶ����)
X1 = fft(x);			% spectrum
% X1 = fftshift(X1);		% put zero freq at the center (�N�W�v�b���s�I�m��)
% Plot spectral magnitude
freq = freqStep*(-N/2:N/2-1);	% freq resolution in spectrum (�W�쪺�W�v���ѪR��)
plot(freq, abs(X1), '.-b'); grid on
xlabel('Frequency(Hz)'); 
ylabel('Magnitude');


