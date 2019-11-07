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


class Sample(object):
        """docstring for Sample"""
        def __init__(self, name,
                     d=0, ns=0, n2=1, B=0, func='Drude', massEffRatio=0.1,
                     N=1e17, tau=100e-15, T=300, pts=1000, fMax=4):
                self.name = name
                self.n = 0
                self.ns = ns
                self.n2 = n2
                self.d = d
                self.massEffRatio = massEffRatio
                self.mass = self.massEffRatio * m0
                self.func = func
                self.HHMass = 1
                self.LHMass = 1
                self.eStatic = 1
                self.eInf = 1
                self.Lattice = 1
                self.OPE = 1
                self.Eg = np.infty
                self.mu_e = 1
                self.mu_h = 1
                self.d_e = 1
                self.d_h = 1
                self.T = T

                if self.name == 'InAs':
                        self.n = 3.51
                        if d == 0:
                                self.d = 500e-9
                        elif d != 0:
                                self.d = d
                        if ns == 0:
                                self.ns = 3.5
                        elif ns != 0:
                                self.ns = ns
                        self.mass = 0.022 * m0
                        self.massEffRatio = 0.022
                        self.func = func
                        self.HHMass = 0.41 * m0
                        self.LHMass = 0.026 * m0
                        self.eStatic = 15.15
                        self.eInf = 12.3
                        self.Lattice = 6.0583
                        self.OPE = 0.030
                        self.Eg = 0.354
                        self.mu_e = 4e4
                        self.mu_h = 5e2
                        self.d_e = 1e3
                        self.d_h = 13
                elif self.name == 'GaAs':
                        self.n = 3.3
                        if d == 0:
                                self.d = 500e-9
                        elif d != 0:
                                self.d = d
                        if ns == 0:
                                self.ns = 3.5
                        elif ns != 0:
                                self.ns = ns
                        self.massEffRatio = 0.063
                        self.mass = 0.063 * m0
                        self.func = func
                        self.HHMass = 0.51 * m0
                        self.LHMass = 0.082 * m0
                        self.eStatic = 12.9
                        self.eInf = 10.89
                        self.Lattice = 5.65325
                        self.OPE = 0.035
                        self.Eg = 1.424
                        self.mu_e = 4e5
                        self.mu_h = 4e2
                        self.d_e = 2e2
                        self.d_h = 10
                elif self.name == 'InSb':
                        self.n = 3.51
                        if d == 0:
                                self.d = 6500e-9
                        elif d != 0:
                                self.d = d
                        if ns == 0:
                                self.ns = 3.5
                        elif ns != 0:
                                self.ns = ns
                        if massEffRatio >= 0.1:
                                self.massEffRatio = 0.014
                                self.mass = 0.014 * m0
                        self.func = func
                        self.HHMass = 0.43 * m0
                        self.LHMass = 0.015 * m0
                        self.eStatic = 16.8
                        self.eInf = 15.7
                        self.Lattice = 6.479
                        self.OPE = 0.025
                        self.Eg = 0.17
                        self.mu_e = 7.7e4
                        self.mu_h = 850
                        self.d_e = 2e3
                        self.d_h = 22
                elif self.name == 'AlInSb':
                        self.n = 3.5
                        if d == 0:
                                self.d = 3000e-9
                        elif d != 0:
                                self.d = d
                        if ns == 0:
                                self.ns = 3.5
                        elif ns != 0:
                                self.ns = ns
                        self.massEffRatio = 0.022
                        self.mass = 0.022 * m0
                        self.func = func
                elif self.name == '3C-SiC':
                        self.n = 2.55
                        if d == 0:
                                self.d = 300e-9
                        elif d != 0:
                                self.d = d
                        if ns == 0:
                                self.ns = 3.45
                        elif ns != 0:
                                self.ns = ns
                        self.mass = 0.25 * m0
                        self.massEffRatio = 0.25
                        self.func = func
                        # self.HHMass = 0.41 * m0
                        # self.LHMass = 0.026 * m0
                        self.eStatic = 9.72
                        self.eInf = 6.52
                        self.Lattice = 4.3596
                        self.OPE = 0.1028
                        self.Eg = 2.39
                        self.mu_e = 400
                        self.mu_h = 50
                        self.d_e = 20
                        self.d_h = 8
                elif self.name == '3C-SiC-Memb':
                        self.n = 2.55
                        if d == 0:
                                self.d = 300e-9
                        elif d != 0:
                                self.d = d
                        if ns == 0:
                                self.ns = 1
                        elif ns != 0:
                                self.ns = ns
                        self.mass = 0.25 * m0
                        self.massEffRatio = 0.25
                        self.func = func
                        # self.HHMass = 0.41 * m0
                        # self.LHMass = 0.026 * m0
                        self.eStatic = 9.72
                        self.eInf = 6.52
                        self.Lattice = 4.3596
                        self.OPE = 0.1028
                        self.Eg = 2.39
                        self.mu_e = 400
                        self.mu_h = 50
                        self.d_e = 20
                        self.d_h = 8
                elif self.name == 'CsSnI3':
                        self.n = 1
                        if d == 0:
                                self.d = 50e-9
                        elif d != 0:
                                self.d = d
                        if ns == 0:
                                self.ns = 2
                        elif ns != 0:
                                self.ns = ns
                        self.mass = 0.2 * m0
                        self.massEffRatio = 0.2
                        self.func = func
                        # self.HHMass = 0.41 * m0
                        # self.LHMass = 0.026 * m0
                        # self.eStatic = 9.72
                        # self.eInf = 6.52
                        # self.Lattice = 4.3596
                        # self.OPE = 0.1028
                        # self.Eg = 2.39
                        # self.mu_e = 400
                        # self.mu_h = 50
                        # self.d_e = 20
                        # self.d_h = 8
                elif self.name == 'MAPI':
                        self.n = 1
                        if d == 0:
                                self.d = 80e-9
                        elif d != 0:
                                self.d = d
                        if ns == 0:
                                self.ns = 2
                        elif ns != 0:
                                self.ns = ns
                        self.mass = 0.2 * m0
                        self.massEffRatio = 0.2
                        self.func = func
                        # self.HHMass = 0.41 * m0
                        # self.LHMass = 0.026 * m0
                        # self.eStatic = 9.72
                        # self.eInf = 6.52
                        # self.Lattice = 4.3596
                        # self.OPE = 0.1028
                        # self.Eg = 2.39
                        # self.mu_e = 400
                        # self.mu_h = 50
                        # self.d_e = 20
                        # self.d_h = 8
                elif self.name == 'GeSn':
                        # self.n = 2.55
                        if d == 0:
                                self.d = 30e-9
                        elif d != 0:
                                self.d = d
                        self.mass = 0.03 * m0
                        self.massEffRatio = 0.03
                        # self.func = func
                        # # self.HHMass = 0.41 * m0
                        # # self.LHMass = 0.026 * m0
                        # self.eStatic = 9.72
                        # self.eInf = 6.52
                        # self.Lattice = 4.3596
                        # self.OPE = 0.1028
                        self.Eg = .6
                        # self.mu_e = 400
                        # self.mu_h = 50
                        # self.d_e = 20
                        # self.d_h = 8
                self.pts = pts
                self.fMax = fMax
                self.B = B
                self.N = N * 1e6
                self.tau = tau
                f = np.linspace(0.1, fMax, pts)

                self.calc(f, self.func)

        def __str__(self):
                return ('\n' +
                        str(self.name) + ':\n' + 'n=' +
                        str(self.n) + ',\nd=' + str(self.d) + '\nn_sub= ' +
                        str(self.ns) + '\nm/m0=' + str(self.massEffRatio) +
                        '\nEg=' + str(self.Eg) + '\n')

        def calc(self, f, model):
                om = f * 2e12 * np.pi
                para2, para3 = 0, 0
                par = fit.Parameters()
                par.add('tau', value=self.tau)
                par.add('N', value=self.N)
                cond0 = self.N * self.tau * e**2 / self.mass
                if model == 'Drude':
                        para = 0
                elif model == 'Cyclotron':
                        para = self.B
                        fC = e * self.B / (self.mass * 2e12 * np.pi)
                        cond0 = self.N * e**2 * self.tau / self.mass
                        par.add('fC', value=fC)
                elif model == 'DrudeNonP':
                        para = self.Eg
                elif model == 'TransThin' or model == 'TransFull':
                        para = self.d
                elif model == 'PhotoTransmission':
                        para = self.d
                        para2 = 650e-15
                        para3 = 1.15e22
                elif model == 'QT':
                        para = self.T
                else:
                        para = 0
                        model = 'Drude'
                        wn.Warn('Warning: model not understood,' +
                                ' Drude is assumed',
                                RuntimeWarning)
                cond = Switch(model, par, f,
                              self.massEffRatio, para, para2, para3)
                eps = self.eInf + 1j * cond / (om * e0)
                nC = np.sqrt(eps)
                T = (1 + self.ns) / (cond * Z0 * self.d + 1 + self.ns)
                omC = e * self.B / self.mass

                self.om = om
                self.f = f
                self.cond = cond
                self.cond0 = cond0
                self.eps = eps
                self.nC = nC
                self.T = T
                self.omC = omC
                self.fC = omC / (2e12 * np.pi)
                self.omP = np.sqrt(self.N * e**2 / (self.mass * e0))
                self.mobility = e * self.tau / self.mass


