[y, fs]=audioread('D:\Master (2019-2021)\Homework\Digital signal processing\DAC\music.wav');
time=(1:length(y))/fs;	% �ɶ��b���V�q
plot(time, y(:,1));	% �e�X�ɶ��b�W���i��
%sound(y,fs);
B=[y(:,1)+1];    % �ݶפJ����ơ]�x�}�Φ��^
xlswrite('D:\Master (2019-2021)\Homework\Digital signal processing\DAC_echo\digital_3.xlsx', B);

