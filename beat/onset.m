clear all;

%% Load audio file

[y, Fs] = wavread('scale.wav');

% only get first channel (two if stereo)
channel = y(:, 1);

%% STFT

% size of each window
windowSize = 1024;

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
    onset = 0;
    for bucket = 1:windowSize
        history = spec(bucket,floor(max(1,i-Fs/windowSize)):i);
        if(spec(bucket,i) > mean(history) + 6*std(history))
            onset = 1;
        end
    end
    if (onset == 1 && lockout < 0)
        fprintf('onset\n')
        processed = [processed i*windowSize/2];
        lockout = 10;
    end
    
    lockout = lockout - 1;
end

%% Mark onsets aurally

output = channel;
lengthS = 1000/44100;
t1 = 0:1:44100*lengthS;
y1 = 3/4*sin((2*pi/(44100/440))*t1);
y2 = y1*4;
st1 = size(t1,2);
for i = 1:size(processed,2)
    for j = 1:st1
        ind = processed(i)+j-250;
        output(ind) = output(ind)*0.25;
        output(ind) = output(ind)+y1(j);
    end
end

%% Write annotated .wav file
wavwrite(output, Fs, 'onsets.wav');

