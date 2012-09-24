clear all;

%% Load audio file

[y, Fs] = wavread('scale.wav');

% only get first channel (two if stereo)
% only get first 15 secs (current onset computation is slow)
channel = y(1:(min(end,Fs*15)),1);

%% Time plot
plot(channel)

%% STFT

% size of each window (can play with this if you want)
windowSize = 2^10;

nWindows = floor(numel(channel)/windowSize*2);
spec = zeros(windowSize,nWindows-2);
for i = 1:nWindows-2
    start = (i-1)*windowSize/2+1;
    window = channel(start:(start+windowSize-1)).*hann(windowSize);
    spec(1:end,i) = abs(fft(window));
end

%% Generate spectra time lapse
for i = 1:nWindows
    plot(spec(i,:))
    pause(windowSize/Fs);
end

%% Stationary spectrum
image(255*spec/max(max(spec)));

%% Compute onsets (very poor algorithm)
processed = [];
lockout = -1;
for i = 1:size(spec,2)
    i
    onset = 0; % guess no onset
    for bucket = 1:windowSize
        history = spec(bucket,floor(max(1,i-Fs/windowSize)):i);
        if(spec(bucket,i) > mean(history) + 6*std(history))
            onset = 1; % detected energy variation > 6sigma from mean
        end
    end
    if (onset == 1 && lockout < 0)
        fprintf('onset\n')
        processed = [processed i*windowSize/2];
        
        % don't let onsets pile up on each other
        lockout = floor(Fs/(windowSize*4))
    end
    
    lockout = lockout - 1;
end

%% Mark onsets aurally

output = channel;
mark = 3/4*sin((2*pi/(44100/440))*(0:1:floor(Fs/44.1)));
for i = 1:size(processed,2)
    for j = 1:numel(mark)
        ind = processed(i)+j-250;
        output(ind) = output(ind)*0.25;
        output(ind) = output(ind)+mark(j);
    end
end

%% Write annotated .wav file
wavwrite(output, Fs, 'onsets.wav');
