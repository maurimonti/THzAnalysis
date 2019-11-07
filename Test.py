import THzAnalysis as thz
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

path = ''
name = ''
dataFiles = [path + name, ]

mes = thz.THzAnalyser(dataFiles, dataFiles, 'abcd', 'CsSnI3', flip=True,
                      stop=0, start=0, plot=False)

f = mes.f
sToT = mes.sigma
ssr = mes.sigmaUncReal
ssi = mes.sigmaUncImag


plt.errorbar(f, np.real(sToT) / 100, ssr / 100,
             ls='', marker='o')

plt.errorbar(f, np.imag(sToT) / 100, ssi / 100,
             ls='', fmt='o', mfc='none')

plt.xlabel('$\\nu$(THz)')
plt.ylabel('$\sigma$(S/cm)')
plt.xlim(0.5, 3)
plt.ylim(-50, 60)

plt.tight_layout()
plt.show()
