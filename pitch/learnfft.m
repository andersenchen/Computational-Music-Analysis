clear all;

%% Train the classifier
[classifier, tclass] = train_joint();

%% Load music here!
[y, Fs] = wavread('chord.wav');
full = zeros(1,size(y,1));
newSize = int32(size(y,1));
for i = 1:newSize
    full(i) = y(i,1);
end

% might not want to load whole thing if it's long
%oneChannel = full(1,1:Fs*20);
oneChannel = full(1,:);

% FFT, processing
ySize = size(oneChannel,2);
windowSize = 2^12;
nWindows = int32(ySize/windowSize*2);
hannWindow = hann(windowSize);
spectrum = zeros(nWindows,windowSize);
truespectrum = zeros(size(spectrum));

for i = 1:nWindows-2
    start = int32((i-1)*windowSize/2)+1;
    myWindow = oneChannel(start:(start+windowSize)-1);
    myWindow1 = myWindow'.*hannWindow;
    truespectrum(i,:) = fft(myWindow1);
    spectrum(i,:) = abs(fft(myWindow1));
end

%% NMF BITCH
tic
spectra = 0;
coeff = 0;
argmin = 0;
min = Inf;

for i = 8
    
    i
    [spectra, coeff] = nmfsc(spectrum' ,i, [], 0.6, 'wat', 0);
    entropy = sum(sum(spectrum'-(spectra*coeff)).^2)/i*sqrt(i);
    if entropy < min
        argmin = i;
        min = entropy;
    end
end
argmin
min


%, 'w0', classifier');
toc
%%
firstNote = spectra(:, 7);
in = ifft(firstNote);
in = repmat(in, 10, 1);
player = audioplayer(in, 44100);
play(player);

%% Pseudoinverse solution

M = pinv(classifier)';

spectra = zeros(nWindows,16);
for i = 1:nWindows
    % normalize to unit vector
    u = spectrum(i,:) / sum(spectrum(i,:));

    % pseudoinverse solution
    spectra(i,:) = M*u';
end

%% NMF solution (multiplicative rule)

spectra = zeros(nWindows,16);
for i = 1:nWindows
    % normalize to unit vector
    u = spectrum(i,:) / sum(spectrum(i,:));

    % NMF
    spectra(i,:) = .5*ones(16,1);
    for k = 1:20 % until convergence
        spectra(i,:) = (spectra(i,:)'.*(classifier*(spectrum(i,:)'./(classifier'*spectra(i,:)')))) ...
                       ./(classifier*ones(size(spectrum(i,:)')));
    end
end

%% gradient descent solution (additive)

spectra = zeros(nWindows,16);
for i = 1:nWindows
    % normalize to unit vector
    u = spectrum(i,:) / sum(spectrum(i,:));
 
    spectra(i,:) = zeros(16,1);
    eps = 50;
    for k = 1:100 % until convergence
        spectra(i,:) = spectra(i,:) + ...
            eps*(classifier*spectrum(i,:)' - classifier*classifier'*spectra(i,:)')';
%        spectra(i,:) = spectra(i,:) + ...
%            eps*(classifier*spectrum(i,:)' - classifier*classifier'*spectra(i,:)' + spectra(i,:)')';

        % constraint enforcement; project back into domain
        for m = 1:size(spectra,2)
            if spectra(i,m) < 0
                spectra(i,m) = 0;
            end
        end
        
        % sparsity
        if k > 50
            for m = 1:size(spectra,2)
                if spectra(i,m) < 300
                    spectra(i,m) = 0;
                end
            end
        end

        if eps > 2
            eps = eps - 1;
        else
            eps = .90*eps;
        end
    end
end

%% Graph output

output = fliplr(spectra); % to print right
surf([coeff' zeros(nWindows,1)]); % need extra padding for surf (wtf)

%% Generate spectra time lapse
for i = 1:nWindows
    plot(spectra(i,:))
    pause(windowSize/Fs);
end

%% Generate an interpretive output (won't work; boundary issue insurmountable)
newChannel = zeros(size(oneChannel));
lastWindow = ifft(tclass'*spectra(1,:)');
max(lastWindow)
min(lastWindow)
for i = 2:nWindows-1
    start = int32((i-1)*windowSize/2)+1;
    myWindow = ifft(tclass'*spectra(i,:)');
    myWindow1 = lastWindow(windowSize/2+1:end) + myWindow(1:windowSize/2);
    newChannel(start:(start+windowSize/2)-1) = myWindow1;
    lastWindow = myWindow;
end

% mastering
newChannel = newChannel / max(newChannel);

% Write .wav file
wavwrite(newChannel, Fs, 'gen.wav');
