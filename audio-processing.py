import numpy as np
import pyaudio
from scipy.io.wavfile import write as write_wav

"""
        Physical definitions (all units SI)
"""

L = 1                # length of string
T = 65               # string tension
mu = 0.005           # linear mass density
v = np.sqrt(T/mu)    # wave speed in massive string
ff0 = 0.5 * v / L    # fundamental frequency

"""
        Store frequencies computed for stiff string vs
        those expected in an ideal string.
"""

numerical_freqs = {1: 1/0.016894, 2: 1/0.007641, 3: 1/0.004455,
                   4: 1/0.002911, 5: 1/0.002030, 6: 1/0.001489,}
ideal_freqs = {1:1*ff0, 2:2*ff0, 3:3*ff0, 4:4*ff0, 
               5:5*ff0, 6:6*ff0,}

"""
        Audio definitions
"""

amp = 0.3  # adjust to change volume
dur = 2 # adjust to lengthen audio
sampling_rate = 44100
t = np.arange(sampling_rate * dur)

"""
        Generate audio for numerical stiff spring frequencies
        and for expect ideal string frequencies side by side
        then write them out to WAV files.
"""

for n, freq in numerical_freqs.items():

        ### Generate waveform from numerical solution of stiff string
        numerical_freq = amp * np.sin(2*np.pi * freq * t / sampling_rate)
        numerical_freq = numerical_freq.astype(np.float32)
        numerical_freq_bytes = numerical_freq.tobytes()

        ### Generate waveform from expected frequency of ideal string
        ideal_freq = amp * np.sin(2*np.pi * ideal_freqs[n] * t / sampling_rate)
        ideal_freq = ideal_freq.astype(np.float32)
        ideal_freq_bytes = ideal_freq.tobytes()

        print("\nn="+str(n))

        """
        NOTE:   Comment out below (highlight and Ctrl-/) to avoid
                playing sounds for each of the frequencies.
        """

        ### Play the sound...
        pa = pyaudio.PyAudio()
        audio = pa.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=sampling_rate,
                        output=True,)
        ### ...first ideal expected
        print("Playing ideal string")
        audio.write(ideal_freq_bytes)
        ### ...then numerical
        print("Playing stiff string")
        audio.write(numerical_freq_bytes)
        ### Close audio stream
        audio.stop_stream()
        audio.close()
        pa.terminate()

        ### NOTE: Above audio code adapted from community at
        ### https://stackoverflow.com/questions/8299303/generating-sine-wave-sound-in-python

        """
        NOTE:   Comment out below (highlight and Ctrl-/) to avoid
                writing out audio to WAV files.
        """

        ### Write out sound to WAV file
        print("Writing out numerical")
        write_wav('numerical_freq'+str(n)+'.wav', sampling_rate, numerical_freq)
        print("Writing out ideal")
        write_wav('ideal_freq'+str(n)+'.wav', sampling_rate, ideal_freq)
