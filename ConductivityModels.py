import numpy as np
import scipy.constants as const
import warnings as wn

import lmfit as fit

e = const.e
m0 = const.m_e
vacImp = const.physical_constants['characteristic impedance of vacuum']
Z0 = vacImp[0]
e0 = const.epsilon_0
c = const.c
hbar = const.hbar
h = const.h
kB = const.Boltzmann


def Switch(name, par, x, paras):
        mapper = {'Drude': Drude,
                  'Cyclotron': Cyclotron,
                  'fC': fC,
                  'DoubleDrude': DoubleDrude,
                  'Line': Line,
                  'QT': QuantumTunneling,
                  'Drude-Smith': Drude_Smith,
                  'CyclotronTransmission': CyclotronTransmission, }
        return mapper[name](par, x, paras)


def Drude(par, x, paras):
        val = par.valuesdict()
        tau = val['tau']
        N = val['N']
        mStarRatio = paras[0]
        mStar = mStarRatio * const.m_e

        sigma = (N * e**2 * tau / mStar) /\
            (1 - x * 2e12j * np.pi * tau)

        return sigma


def Cyclotron(par, x, B):
        # B = paras[1]
        val = par.valuesdict()
        sigma0 = val['N']
        tau = val['tau']
        omegaC = val['fC'] * 1e12 * 2 * np.pi

        # sigma = (N * e**2 * tau / (m0 * mStarRatio)) *\
        #         (1 - 1j * x * 1e12 * 2 * np.pi * tau) /\
        #         ((1 - 1j * 2 * np.pi * x * tau * 1e12)**2 +
        #          (e * B * tau / (m0 * mStarRatio))**2)
        sigma = sigma0 * (1 - 1j * 2 * np.pi * x * tau * 1e12) /\
            ((1 - 1j * 2 * np.pi * x * tau * 1e12)**2 + omegaC**2 * tau**2)
        return sigma


def CyclotronTransmission(par, x, paras):
        val = par.valuesdict()
        A = val['A']
        gamma = val['gamma'] * 2e12 * np.pi
        omegaC = val['fC'] * 2e12 * np.pi
        omega = x * 2e12 * np.pi

        L = A * 0.5 * gamma / ((omega - omegaC)**2 + (0.5 * gamma)**2)
        T = 1 - L
        return T


def Drude_Smith(par, x, paras):
        val = par.valuesdict()
        tau = val['tau']
        N = val['N']
        c1 = val['c1']
        mr = paras['mr']

        mStar = mr * const.m_e

        sigma = ((N * e**2 * tau / mStar) /
                 (1 - x * 2e12j * np.pi * tau)) * (1 + c1 /
                                                   (1 -
                                                    2e12j * np.pi * x * tau))
        return sigma


def ColeDavidson(par, x, paras):
        val = par.valuesdict()
        N = val['N']
        tau = val['tau']
        b = val['b']
        mr = paras['mr']

        mStar = mr * const.m_e

        sigma = (N * e**2 * tau / mStar) / (1 - 2e12j * np.pi * x * tau)**b
        return sigma


def Lorentz(par, x):
        val = par.valuesdict()
        A = val['A']
        gamma = val['gamma']
        f0 = val['f0']
        om = 2e12 * np.pi * x
        om0 = f0 * np.pi * 2e12

        L = A * 0.5 * gamma / ((om - om0)**2 + (0.5 * gamma)**2)
        return L


def Line(par, x, paras):
        val = par.valuesdict()
        A = val['A']
        B = val['B']

        line = A * x + B
        return line


