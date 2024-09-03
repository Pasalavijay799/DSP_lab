
from scipy.io import wavfile

fs, x = wavfile.read('whisper-trail-2ogg-14429.wav')


def sampling(x, a):
    if a > 1:
        y=x[::a]
        wavfile.write('ds1.wav',fs,y)


a=int(input("Enter sampling factor:"))
sampling(x, a)
