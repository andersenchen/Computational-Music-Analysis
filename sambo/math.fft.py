
# # # sine wave <= math
times = arange(1024) # |samples| = 2^x
spectrum = abs( fft(sin(times)) ) # 440Hz?
freqs    = fftfreq(times.shape[-1])
#periods  = 1/freqs
#semilogx( half(freqs), half(spectrum) ); show()

""" fftfreq(n=window_length, d=sample_spacing)
fftfreq(8) = [0/8, 1/8, 2/8, 3/8,  -4/8, -3/8, -2/8, -1/8]
half(fftfreq(n:even)) = [ 0/m 1/m .. (m-1)/m ] where m = n/2
"""
