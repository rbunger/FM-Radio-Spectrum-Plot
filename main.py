from numpy import argsort, zeros, arctan2, conj
from rtlsdr import RtlSdr
from numpy.fft import fft
import matplotlib.pyplot as plt

kHz = 1e3
MHz = 1e6


def FourierSampling(N, sample_rate):
    dt = 1 / sample_rate
    df = 1 / (N * dt)
    t = zeros(N)
    f = zeros(N)
    for j in range(-N // 2, N // 2):
        t[j] = j * dt
        f[j] = j * df
    return t, f, dt


sdr = RtlSdr()

N = 2 ** 14
B = 120 * kHz
sdr.sample_rate = 2.5 * MHz
sdr.center_freq = 101.2 * MHz
sdr.gain = 20

t, f, dt = FourierSampling(N, sdr.sample_rate)

iq = sdr.read_samples(2048 + N)[2048:]
sdr.close()

s = zeros(N)
for j in range(1, N):
    q = iq[j] * conj(iq[j - 1])
    s[j] = arctan2(q.imag, q.real)

S = fft(s)

S_max = 0
for j in range(-N // 2, N // 2):
    if (f[j] >= -B / 2 and f[j] <= B / 2):
        if (abs(S[j]) > S_max):
            S_max = abs(S[j])

o = argsort(t)

plt.figure("IQ baseband signal")
plt.plot(t[o], iq[o].real, label="real")
plt.plot(t[o], iq[o].imag, label="imag")
plt.xlabel("t / s")
plt.grid()
plt.legend()

plt.figure("Phase-demodulated signal")
plt.plot(t[o], s[o])
plt.xlabel("t / s")
plt.grid()

plt.figure("Spectrum of phase-demodulated signal")
plt.xlim([-B / 4 / kHz, B / 2 / kHz])
plt.plot(f[o] / kHz, abs(S[o]) / S_max)
plt.xlabel("f / kHz")
plt.grid()
plt.ylim([-0.1, 1.1])
plt.axvspan(0, 15, facecolor="red", alpha=0.1)
plt.annotate('L+R', xy=(5, 0.925))
plt.axvspan(23, 37.5, facecolor="green", alpha=0.1)
plt.annotate('L-R', xy=(28, 0.925))
plt.axvspan(38.5, 53, facecolor="green", alpha=0.1)
plt.annotate('L-R', xy=(43.1, 0.925))

plt.show()
