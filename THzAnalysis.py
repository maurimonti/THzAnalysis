import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cnst
import warnings as wn

import lmfit as fit

import src.data_manipulation_functions as man
import src.math_functions as mt
import ConductivityModels as mod
import Sample as sam


class THzAnalyser(object):
        """Class which defines an THzAnalyser object which is
        used to analyze typical data from a THZ-TDS experiment

        ...

        Attributes
        ----------
        fileList: list of str
            names of the files to be analysed. They can be repetition of the
            same experiment or the result of different experiment.
        refList: list of str
            names of the file to be used as reference. Can be identical to
            fileList if that's what the data format requires (see fmt)
        fmt: str
            format of the stored data, three values accepted: 'Ox', 'TW'
            and 'abcd'. more info in the readme
        sampleName: str
            name of the sample measured, if the sample name is not recognized
            the values for GaAs will be assumed. This will lead to errors in
            the conductivity and mobility absolute values. Although the
            frequency response will be unaffected. See the readme for how
            to write a new sample model.
        d: float
            thickness of the sample, default 0. If 0 an arbitrary default value
            for the indicated sample will be assumed. If no recognized sample
            was given, the value will be the default of GaAs indicated in the
            readme.
        ns: float
            substrate refractive index, default 0. If 0 an arbitrary default
            value for the indicated sample will be assumed. If no recognized
            sample was given, the value will be the default of GaAs indicated
            in the readme.
        n2: float
            refractive index of the encapsulation layer or medium in which the
            THz was travelling prior of hitting the sample. Default 1 (air)
        exp: int
            zero padding factor if greater than zero increases the time window
            of exp * current time window
        start: int
            index of the first TDS time point to consider, default 0
            (from first point)
        stop: int
            index of the last TDS time point to consider, measured from the
            last index, default 0.
        flip: bool
            if True flips the time axis, default False
        plot: bool
            if True generates some automatic plots, default True
        window: str
            the type of window to use to window the time resolved data,
            default '', none used. Only accepted windows are Gaussian and
            Lorentz shapes
        windowpara: list of float
            parameters of the desired window, if none given unit values will
            be assumed
        fitFlag: bool
            if True performs a fit on the averaged spectra, default False
        multipleFit: bool
            if True considers the different files as coming from different
            experiments and will fit them separately. If False will fit the
            average of the files provided. Default False.
        model: str
            model used to fit the spectra, more details in the readme.
        complexFunction: bool
            If True will attempt to fit the complex valued spectra.
            Default False
        init: dictionary or list of dictionary
            dictionary containing the initial guesses for the fit parameters,
            if multipleFit is True provide a list of dictionaries, one for each
            experiment
            default empty (default values will be used)
        para: list of float
            list of possible external parameters that might be needed for the
            fit, ex. magnetic field for cyclotron
            default empty list []
        boundaries: list of int or list of lists of int
            list of the frequency values between which the fit has to be
            performed. Unfortunately only in data point index.
            if multipleFit is True provide a list of lists, one for each
            experiment
            default All spectrum considered
        guess: bool
            if plot is True, controls the plotting of the initial guess of the
            fit parameters, useful for complicated fits
            Default False
        fitQty: str
            Quantity to be fitted, allowed values are 'Conductivity' and
            'Transmission'
            Default 'Conductivity'
        thin: bool
            if True assumes a thin film samples. Not properly implemented.
            More information in the readme.
            default True

        Methods
        -------
        Data_Reader(fileList, fmt, col=0, shape=0)
            Reads the data files
        Data_Computation(E, ERef, x, xRef,
                             sample, fmt, flip=False, exp=0,
                             window='', para=[], thin=True)
            Computes all the spectra and quantities connected
        Data_Plotter(self, fmt)
            A simple automatic plotting routine
        Fit(x, y, err=0, model='', init=0,
                        para=0, c=False, plot=False, guess=False,
                        fitQty='Conductivity')
            Performs fits on the computed spectra

        """
        def __init__(self, fileList, refList,
                     fmt, sampleName, d=0, ns=0, n2=1,
                     exp=0, start=0, stop=0, flip=False, plot=True,
                     window='', windowPara=[],
                     fitFlag=False, multipleFit=False, model='',
                     complexFunction=False,
                     init=[], para=[], boundaries=[0, 1000], guess=False,
                     fitQty='Conductivity', thin=True):
                """
                Parameters
                ----------
                fileList: list of str
                    names of the files to be analysed.
                    They can be repetition of the same experiment or
                    the result of different experiment.
                refList: list of str
                    names of the file to be used as reference.
                    Can be identical to fileList if that's what the data
                    format requires (see fmt)
                fmt: str
                    format of the stored data, three values accepted:
                    'Ox', 'TW' and 'abcd'. more info in the readme
                sampleName: str
                    name of the sample measured, if the sample name is not
                    recognized the values for GaAs will be assumed.
                    This will lead to errors in the conductivity and mobility
                    absolute values. Although the frequency response will be
                    unaffected. See the readme for howto write a
                    new sample model.
                d: float
                    thickness of the sample, default 0. If 0 an arbitrary
                    default value for the indicated sample will be assumed.
                    If no recognized sample was given, the value will be
                    the default of GaAs indicated in the readme.
                ns: float
                    substrate refractive index, default 0. If 0 an arbitrary
                    default value for the indicated sample will be assumed.
                    If no recognized sample was given, the value will be
                    the default of GaAs indicated in the readme.
                n2: float
                    refractive index of the encapsulation layer or medium
                    in which the THz was travelling prior of hitting
                    the sample. Default 1 (air)
                exp: int
                    zero padding factor if greater than zero increases the
                    time window of exp * current time window
                start: int
                    index of the first TDS time point to consider, default 0
                    (from first point)
                stop: int
                    index of the last TDS time point to consider,
                    measured from the last index, default 0.
                flip: bool
                    if True flips the time axis, default False
                plot: bool
                    if True generates some automatic plots, default True
                window: str
                    the type of window to use to window the time resolved data,
                    default '', none used. Only accepted windows are
                    Gaussian and Lorentz shapes
                windowpara: list of float
                    parameters of the desired window, if none given
                    unit values will be assumed
                fitFlag: bool
                    if True performs a fit on the averaged spectra,
                    default False
                multipleFit: bool
                    if True considers the different files as coming from
                    different experiments and will fit them separately.
                    If False will fit the average of the files provided.
                    Default False.
                model: str
                    model used to fit the spectra, more details in the readme.
                complexFunction: bool
                    If True will attempt to fit the complex valued spectra.
                    Default False
                init: dictionary or list of dictionary
                    dictionary containing the initial guesses for the fit
                    parameters, if multipleFit is True provide a list of
                    dictionaries, one for each experiment
                    default empty (default values will be used)
                para: list of float
                    list of possible external parameters that might be
                    needed for the fit, ex. magnetic field for cyclotron
                    default empty list []
                boundaries: list of int or list of lists of int
                    list of the frequency values between which the fit has
                    to be performed. Unfortunately only in data point index.
                    if multipleFit is True provide a list of lists,
                    one for each experiment
                    default All spectrum considered
                guess: bool
                    if plot is True, controls the plotting of the initial
                    guess of the fit parameters, useful for complicated fits
                    Default False
                fitQty: str
                    Quantity to be fitted, allowed values are 'Conductivity'
                    and 'Transmission'
                    Default 'Conductivity'
                thin: bool
                    if True assumes a thin film samples.
                    Not properly implemented.
                    More information in the readme.
                    default True

                """
                super(THzAnalyser, self).__init__()
                self.fileList = fileList
                self.window = window
                self.para = para
                self.windowPara = windowPara
                self.sample = sam.Sample(sampleName, d=d, ns=ns, n2=n2)
                self.params = 0
                self.multiParams = []

                if fmt == 'Ox':
                        sigCol = 2
                        refCol = 1
                elif fmt == 'TW':
                        sigCol = 1
                        refCol = 7
                elif fmt == 'abcd':
                        sigCol = 1
                        refCol = 2
                else:
                        sigCol = 1
                        refCol = 1

                shapeData = self.Data_Reader(fileList[0], fmt, 0, shape=0)
                shape = np.shape(shapeData)
                lenFiles = shape[1]
                numFiles = len(fileList)
                lenfft = np.int(np.round((exp + 1) *
                                         (lenFiles -
                                          stop - start) / 2 + .6))
                listShape = ((numFiles,
                              (exp + 1) * (lenFiles -
                                           stop - start)))
                lenT = listShape[1]
                listShapeFft = ((numFiles, lenfft))

                # Time domain arrays
                self.xList = np.zeros(listShape)  # time delay in mm
                self.xRefList = np.zeros(listShape)

                self.tList = np.zeros(listShape)  # time delay in ps
                self.tRefList = np.zeros(listShape)

                self.EtList = np.zeros(listShape)  # Time domain dields
                self.EtRefList = np.zeros(listShape)

                # Averaged time domain arrays
                self.t = np.zeros(lenT)  # Same quantities but averaged
                self.Et = np.zeros(lenT)
                self.EtRef = np.zeros(lenT)

                # Frequency domain arrays
                self.fList = np.zeros(listShapeFft,
                                      dtype=np.complex_)  # Frequency
                self.EList = np.zeros(listShapeFft,
                                      dtype=np.complex_)  # Complex FFT field
                self.ERefList = np.zeros(listShapeFft,       # Complex FFT
                                         dtype=np.complex_)  # Reference field
                self.transList = np.zeros(listShapeFft,
                                          dtype=np.complex_)  # Transmission
                self.sigmaList = np.zeros(listShapeFft,
                                          dtype=np.complex_)  # Conductivity
                self.epsilonList = np.zeros(listShapeFft,       # Dielectric
                                            dtype=np.complex_)  # function
                self.lossList = np.zeros(listShapeFft,       # Loss
                                         dtype=np.complex_)  # Function
                self.nList = np.zeros(listShapeFft,       # Complex refractive
                                      dtype=np.complex_)  # index

                # Averaged frequency domain arrays
                self.f = np.zeros(lenfft,             # Same quantities
                                  dtype=np.complex_)  # but averaged
                self.E = np.zeros(lenfft,
                                  dtype=np.complex_)
                self.ERef = np.zeros(lenfft,
                                     dtype=np.complex_)
                self.sigma = np.zeros(lenfft,
                                      dtype=np.complex_)
                self.trans = np.zeros(lenfft,
                                      dtype=np.complex_)
                self.epsilon = np.zeros(lenfft,
                                        dtype=np.complex_)
                self.loss = np.zeros(lenfft,
                                     dtype=np.complex_)
                self.n = np.zeros(lenfft,
                                  dtype=np.complex_)

                # Uncertainties of frequency domain quantities
                self.sigmaUncReal = np.zeros(lenfft)
                self.sigmaUncImag = np.zeros(lenfft)
                self.transUnc = np.zeros(lenfft)
                self.epsilonUncReal = np.zeros(lenfft)
                self.epsilonUncImag = np.zeros(lenfft)
                self.lossUnc = np.zeros(lenfft)
                self.nUnc = np.zeros(lenfft)

                # self.DrudeCoeff = 0
                stop = lenFiles - stop
                if stop == 0:
                        stop = int(1e6)

                tmp_xList, tmp_EtList = self.Data_Reader(fileList, fmt,
                                                         sigCol, shape)
                tmp_xRefList, tmp_EtRefList = self.Data_Reader(refList, fmt,
                                                               refCol, shape)

                # If # ref < # files puts the first ref as the missing files
                if np.abs(len(refList) - len(fileList)) != 0:
                        for i in range(len(fileList)):
                                tmp_xRefList[i] = tmp_xRefList[0]
                                tmp_EtRefList[i] = tmp_EtRefList[0]

                for i, file in enumerate(fileList):
                        (self.tList[i],
                         self.EtList[i],
                         self.tRefList[i],
                         self.EtRefList[i],
                         self.fList[i],
                         self.EList[i],
                         self.ERefList[i],
                         self.transList[i],
                         self.sigmaList[i],
                         self.epsilonList[i],
                         self.lossList[i],
                         self.nList[i]
                         ) = self.Data_Computation(
                             tmp_EtList[i, start:stop],
                             tmp_EtRefList[i, start:stop],
                             tmp_xList[i, start:stop],
                             tmp_xRefList[i, start:stop],
                             self.sample,
                             fmt,
                             flip=flip,
                             exp=exp,
                             window=window,
                             para=windowPara,
                             thin=thin)
                for i in range(lenT):
                        self.t[i] = np.average(self.tList[:, i])
                        self.Et[i] = np.average(self.EtList[:, i])
                        self.EtRef[i] = np.average(self.EtRefList[:, i])
                for i in range(lenfft):
                        self.f[i] = np.average(self.fList[:, i])
                        self.E[i] = np.average(self.EList[:, i])
                        self.ERef[i] = np.average(self.ERefList[:, i])
                        self.trans[i] = np.average(self.transList[:, i])
                        self.sigma[i] = np.average(self.sigmaList[:, i])
                        self.epsilon[i] = np.average(self.epsilonList[:, i])
                        self.loss[i] = np.average(self.lossList[:, i])
                        self.n[i] = np.average(self.nList[:, i])

                        self.sigmaUncReal[i] = np.std(
                            np.real(self.sigmaList[:, i]))
                        self.sigmaUncImag[i] = np.std(
                            np.imag(self.sigmaList[:, i]))
                        self.transUnc[i] = np.std(np.abs(self.transList[:, i]))
                        self.epsilonUncReal[i] = np.std(
                            np.real(self.epsilonList[:, i]))
                        self.epsilonUncImag[i] = np.std(
                            np.imag(self.epsilonList[:, i]))
                        self.lossUnc[i] = np.std(np.abs(self.lossList[:, i]))
                        self.nUnc[i] = np.std(np.abs(self.nList[:, i]))
                # for i in range(3, lenfft):
                #         tmp = np.sqrt(np.real(self.sigma[i])**2) \
                #             / np.sqrt(np.real(self.sigma[i])**2 +
                #                       np.imag(self.sigma[i])**2)
                #         self.DrudeCoeff += tmp
                #         if np.abs(self.f[i] - 2.5) < 0.03:
                #                 self.DrudeCoeff /= (i - 3)
                #                 break
                # self.ratio = 1e15 * np.imag(self.sigma) / (np.abs(self.f) *
                #                                            2e12 * np.pi *
                #                                            np.real(self.sigma))
                if fitFlag:
                        y_Map = {'Conductivity': self.sigma,
                                 'Transmission': self.trans}
                        if multipleFit:
                            y_Map = {'Conductivity': self.sigmaList,
                                     'Transmission': self.transList}
                        y = y_Map[fitQty]
                        err_Map = {'Conductivity':
                                   [self.sigmaUncReal,
                                    self.sigmaUncImag],
                                   'Transmission':
                                   self.transUnc}
                        err = err_Map[fitQty]
                        if model == '':
                                model = self.sample.f
                                wn.warn('Warning:: model undefined, sample' +
                                        '\'s default chosen: ' +
                                        model, RuntimeWarning)
                        if multipleFit:
                            for i, o in enumerate(y):
                                self.multiParams.append(
                                    self.Fit(x=self.fList[i, boundaries[i][0]:
                                                          boundaries[i][1]],
                                             y=o[boundaries[i][0]:
                                                 boundaries[i][1]],
                                             model=model,
                                             err=0,
                                             init=init[i], para=para[i],
                                             c=complexFunction, guess=guess,
                                             plot=plot,
                                             fitQty=fitQty))
                        elif not multipleFit:
                            self.params = self.Fit(x=self.f[boundaries[0]:
                                                            boundaries[1]],
                                                   y=y[boundaries[0]:
                                                       boundaries[1]],
                                                   model=model,
                                                   err=err[boundaries[0]:
                                                           boundaries[1]],
                                                   init=init, para=para,
                                                   c=complexFunction,
                                                   guess=guess,
                                                   plot=plot,
                                                   fitQty=fitQty)
                if plot:
                        (self.multifig,
                         self.multitimefig,
                         self.finalfig) = self.Data_Plotter(fmt)

                        self.valuesfig = 0

        def __str__(self):
                return (str(self.fileList) + ' ' +
                        str(self.sample) + ' ' +
                        str(self.params))

        @staticmethod
        def Data_Reader(fileList, fmt, col=0, shape=0):
                """
                Reads the text files provided

                The read data are returned as arrays

                Parameters
                ----------
                fileList: list of str
                    List of names of the data files to read
                fmt: str
                    Format the text files are written in, as in which column
                    is the x and which the y
                col: int
                    If shape is nonzero, which column has to be returned as y.
                    Default 0 (which is very likely an x)
                shape: int
                    If zero, returns the whole read data as it is.
                    Default 0 (assumes only 1 file given at the time)

                """
                sr = 0
                delm = '\t'

                if fmt == 'TW':
                        sr = 3
                        delm = ','

                if shape == 0:
                        data = man.Reader(fileList,
                                          delimiter=delm,
                                          skipRows=sr,
                                          caller='')
                        return data
                elif shape != 0:
                        data = np.zeros((len(fileList), shape[0], shape[1]))

                        for i, file in enumerate(fileList):
                                data[i] = man.Reader(file,
                                                     delimiter=delm,
                                                     skipRows=sr,
                                                     caller='')

                        return data[:, 0], data[:, col]

        @staticmethod
        def Data_Computation(E, ERef, x, xRef,
                             sample, fmt='', flip=False, exp=0,
                             window='', para=[], thin=True):
                """
                Computes all the spectral quantities

                Parameters
                ----------
                E: numpy array
                    The time domain electric field pulse
                ERef: numpy array
                    The reference time domain electric field pulse
                x: numpy array
                    The THz-gate time delay, in stage distance (mm)
                xRef: numpy array
                    The THz-gate time delay of the reference,
                    in stage distance (mm)
                sample: sample object
                    The sample investigated
                fmt: str
                    Format the data were written in, changes the electric
                    fields definitions
                flip: bool
                    if True flips the time axis
                    Default False
                exp: int
                    Zero-padding factor
                    Default 0
                window: str
                    The type of window to use to window the data (not well
                    implemented yet)
                    Default '', does nothing
                para: list
                    List of parameters characterising the window.
                    Default [], units values assumed
                thin: bool
                    if True assumes the sample is a thin film with the
                    consequent simplifications
                Default True

                Raises
                ------
                RuntimeWarning
                    If no format is specified it automatically assumes 'abcd'

                """
                vacImp = cnst.physical_constants['characteristic impedance' +
                                                 ' of vacuum']
                Z0 = vacImp[0]  # the value of z0
                e0 = cnst.epsilon_0
                u = 'mm'
                cv = 0.1499
                ns = sample.ns
                n2 = sample.n2
                n = sample.n
                d = sample.d
                eInf = sample.eInf
                M = np.amax(ERef)
                idx = np.where(ERef == M)[0][0]
                shift = x[idx]
                x -= shift
                xRef -= shift
                if flip:
                        E = np.flipud(E)
                        ERef = np.flipud(ERef)
                        # x = np.flipud(x)
                        # xRef = np.flipud(xRef)

                if fmt == 'Ox':
                        (ERef, E) = (ERef - 0.5 * E, ERef + 0.5 * E)
                        x = x - 24
                        xRef = xRef - 24
                elif fmt == 'Wa':
                        E = ERef - E
                        ERef = ERef
                elif fmt == 'TW':
                        E = E
                        ERef = ERef
                        u = 'OD'
                        cv = 0.2998
                elif fmt == 'abcd':
                        pass
                else:
                        pass
                        wn.warn('Warning:: undefined or wrong format, ' +
                                'default one chosen: abcd',
                                RuntimeWarning)
                if window != '':
                        x, E, = man.Window(window,
                                           [p * cv for p in para], x, E)
                        xRef, ERef, = man.Window(window,
                                                 [para[0] * cv, para[1] * cv],
                                                 xRef, ERef)
                if exp > 0:
                        x, E = mt.zeropad(x, E, exp)
                        xRef, ERef = mt.zeropad(xRef, ERef, exp)

                freq, Efft = mt.IFFT(x, E, u)
                freq, EReffft = mt.IFFT(xRef, ERef, u)

                t = x / cv
                tRef = xRef / cv

                trans = Efft / EReffft
                if thin:
                        sigma = -(ns + n2) * (Efft -
                                              EReffft) / (Z0 * d * EReffft)
                elif not thin:
                        dEonE = (Efft - EReffft) / EReffft
                        # sigma = -(2 / (Z0 * d) - 2e12j * freq * np.pi * e0 *
                        #           (n**2 + 1)) * dEonE / (1 + dEonE)
                        sigma = -(2 / (Z0 * d)) * dEonE / (1 + dEonE)

                epsilon = eInf + 1j * sigma / (freq * 2e12 * np.pi * e0)

                loss = np.imag(-1 / epsilon)

                n = np.sqrt(epsilon)
                return (t, E, tRef, ERef,
                        freq, Efft, EReffft,
                        trans, sigma, epsilon, loss, n)

        def Data_Plotter(self, fmt=''):
                """
                Default plots some results

                Parameters
                ----------
                fmt: str
                    Format the data were saved in, needed because changes
                    some quantities definitions

                Raises
                ------
                RuntimeWarning
                    If no format is specified it automatically assumes 'abcd'

                """
                fMin = 0.2
                fMax = 2
                fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2,
                                                             sharex='all',
                                                             squeeze=True)
                for s in self.sigmaList:
                        i = np.where(s == self.sigmaList)[0][0]
                        (idxMin,
                         idxMax,
                         reCMin,
                         reCMax) = man.Extrema(fMin, fMax,
                                               self.fList[i],
                                               np.array(
                                                   (np.real(
                                                       s),)))
                        (idxMin,
                         idxMax,
                         imCMin,
                         imCMax) = man.Extrema(fMin, fMax,
                                               self.fList[i],
                                               np.array((np.imag(s),)))
                        plot0 = ax0.plot(np.real(self.fList[i]),
                                         np.real(s), ls='-')
                        color = plot0[0].get_color()
                        ax0.plot(np.real(self.fList[i]),
                                 np.imag(s), ls='--', color=color)
                        ax0.set_ylim(min(reCMin, imCMin), max(reCMax, imCMax))

                        (idxMin,
                         idxMax,
                         reCMin,
                         reCMax) = man.Extrema(fMin, fMax,
                                               self.fList[i],
                                               np.array(
                                                   (np.real(
                                                       self.epsilonList[i]),)))
                        (idxMin,
                         idxMax,
                         imCMin,
                         imCMax) = man.Extrema(fMin, fMax,
                                               self.fList[i],
                                               np.array(
                                                   (np.imag(
                                                    self.epsilonList[i]),)))
                        plot1 = ax1.plot(np.real(self.fList[i]),
                                         np.real(self.epsilonList[i]), ls='-')
                        color = plot1[0].get_color()
                        ax1.plot(np.real(self.fList[i]),
                                 np.imag(self.epsilonList[i]),
                                 ls='--', color=color)
                        ax1.set_ylim(min(reCMin, imCMin), max(reCMax, imCMax))
                        (idxMin,
                         idxMax,
                         reCMin,
                         reCMax) = man.Extrema(fMin, fMax,
                                               self.fList[i],
                                               np.array(
                                                   (np.abs(
                                                       self.transList[i]),)))
                        ax2.plot(np.real(self.fList[i]),
                                 np.abs(self.transList[i]), ls='-')
                        ax2.set_ylim(reCMin, reCMax)
                        # (idxMin,
                        #  idxMax,
                        #  reCMin,
                        #  reCMax) = man.Extrema(fMin, fMax,
                        #                        self.fList[i],
                        #                        np.array(
                        #                            ((
                        #                                self.lossList[i]),)))
                        # ax3.plot(self.f, (self.lossList[i]), ls='-')
                        # ax3.set_ylim(reCMin, reCMax)
                        if fmt == 'TW':
                                specToPlotOFF = np.abs(self.ERefList[i] *
                                                       1j * self.fList[i])
                                specToPlotON = np.abs(self.Elist[i] *
                                                      1j * self.fList[i])
                        if fmt == '':
                                wn.warn('Warning:: undefined or wrong' +
                                        ' format, default one chosen: abcd',
                                        RuntimeWarning)
                                specToPlotOFF = np.abs(self.ERefList[i])
                                specToPlotON = np.abs(self.EList[i])
                        elif fmt != 'TW':
                                specToPlotOFF = np.abs(self.ERefList[i])
                                specToPlotON = np.abs(self.EList[i])
                        ax3.semilogy(np.real(self.fList[i]),
                                     specToPlotOFF * 1e3,
                                     color=color, ls='--', label=str(i))
                        ax3.semilogy(np.real(self.fList[i]),
                                     specToPlotON * 1e3,
                                     color=color, ls='-')
                        ax3.set_ylim(1e-4, 1)
                ax0.set_xlim(fMin, fMax)
                ax0.set_ylabel('$\sigma$')
                ax1.yaxis.set_label_position("right")
                ax1.yaxis.tick_right()
                ax1.set_ylabel('$\\varepsilon$')
                ax2.set_ylabel('T')
                ax2.set_xlabel('$\\nu$(THz)')
                # ax3.set_ylabel('Im{$\\frac{-1}{\\varepsilon}$}')
                ax3.set_ylabel('$E_{\mathrm{THz}}(a.u.)$')
                ax3.yaxis.set_label_position("right")
                ax3.yaxis.tick_right()
                ax3.set_xlabel('$\\nu$(THz)')
                plt.tight_layout()
                ax3.legend(loc='best')

                figt, ax0 = plt.subplots(nrows=1, ncols=1)
                for e in self.EtList:
                        i = np.where(e == self.EtList)[0][0]
                        plot0 = ax0.plot(self.tList[i],
                                         e / max(self.EtRefList[i]), ls='-',
                                         label=str(i))
                        color = plot0[0].get_color()
                        ax0.plot(self.tRefList[i],
                                 self.EtRefList[i] / max(self.EtRefList[i]),
                                 color=color, ls='--')
                        ax0.plot(self.tRefList[i],
                                 (self.EtRefList[i] -
                                  e) / max(self.EtRefList[i]),
                                 color=color, ls='-.')
                if self.window != '':
                        ax0.plot(self.t,
                                 mt.Function('Gauss', self.para,
                                             self.t))
                ax0.set_ylabel('E$_{\mathrm{THz}}$(a.u.)')
                ax0.set_xlabel('t(ps)')
                ax0.legend(loc='best')
                finalfig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2,
                                                    sharex='all',
                                                    squeeze=True)

                (idxMin,
                 idxMax,
                 reCMin,
                 reCMax) = man.Extrema(fMin, fMax,
                                       self.f,
                                       np.array(
                                           (np.real(
                                               self.sigma),)))
                (idxMin,
                 idxMax,
                 imCMin,
                 imCMax) = man.Extrema(fMin, fMax,
                                       self.f,
                                       np.array((np.imag(self.sigma),)))
                ax0.errorbar(np.real(self.f), np.real(self.sigma),
                             self.sigmaUncReal, marker='o', ls='')
                ax0.set_xlim(fMin, fMax)
                ax0.set_ylim(reCMin, reCMax)
                ax0.set_xlabel('$\\nu$(THz)')
                ax0.set_ylabel('$\mathcal{Re}\{\sigma\}$(S/m)')

                ax1.errorbar(np.real(self.f), np.imag(self.sigma),
                             self.sigmaUncImag, marker='o', ls='')
                ax1.set_ylim(imCMin, imCMax)
                ax1.set_xlabel('$\\nu$(THz)')
                ax1.set_ylabel('$\mathcal{Im}\{\sigma\}$(S/m)')
                ax1.yaxis.set_label_position("right")
                ax1.yaxis.tick_right()
                plt.tight_layout()

                return fig, figt, finalfig

        @staticmethod
        def Fit(x, y, err=0, model='', init=0,
                para=0, c=False, plot=False, guess=False,
                fitQty='Conductivity'):
                """
                Fits the spectra with the model provided

                Parameters
                ----------
                x: numpy array
                    x value array (ususally frequency)
                y: numpy array
                    y value array, ones to be fitted
                err: 2xN numpy array
                    Real and imaginary uncertainties to be considered in
                    the fit. Has to be a 2xN size array
                    Default 0, no errors considered
                model: str
                    String containing the name of the model to be used for
                    the fit. To check available models and how to write more
                    see the readme.
                    Default '', Drude model assumed
                init: dictionary
                    Initial guess of the parameters in a dictionary
                    To check which names correspond to which parameters see
                    the readme.
                    Default 0, some predetermined arbitrary values are assumed
                para: list
                    Some parameters that can be used in models, e.g. magnetic
                    field in cyclotron spectroscopy
                    Default 0, no parameters used
                c: bool
                    If True fits the complex data
                    Default False
                plot: bool
                    If True automatically plots some of the data and fits
                    Default False
                guess: bool
                    If True plots the model corresponding to the initial
                    guess of the parameters
                    Default False
                fitQty: str
                    Quantity to be fitted, usually 'Conductivity' or
                    'Transmission'
                    Default 'Conductivity'

                Raises
                ------
                RuntimeWarning
                    If model is not recognized, assumes a Drude model with
                    default parameters and raises a warning

                """
                def ResWrap(f, paras, c=False):
                        def Residual(par, x, data=None):
                                model = mod.Switch(f, par, x, paras)
                                if not c:
                                        model = np.real(model)
                                if data is None:
                                        return model
                                dataShape = np.shape(data)

                                resid = model - data
                                if dataShape[0] <= 3:
                                        resid = model - data[0]
                                        err = data[1]
                                        resid = np.sqrt(resid**2 / err**2)
                                return resid.view(np.float)
                        return Residual
                yLabel_Map = {'Conductivity': '$\sigma$',
                              'Transmission': 'T'}
               yLabel = yLabel_Map[fitQty]
                if c:
                        err = err[0] + 1j * err[1]

                elif not c:
                        x, y = np.real(x), np.real(y)
                data = y
                par = fit.Parameters()
                if init:
                        if model == 'Drude':
                                par.add('tau', init['tau'])
                                par.add('N', init['N'] * 1e6)
                                paras = [para['mr']]
                        elif model == 'Cyclotron':
                                par.add('tau', init['tau'])
                                par.add('N', init['sigma0'])
                                par.add('fC', init['fC'])
                                paras = [0, para['B']]
                        elif model == 'CyclotronTransmission':
                                par.add('A', init['A'])
                                par.add('gamma', init['gamma'])
                                par.add('fC', init['fC'])
                                paras = []
                        elif model == 'DrudeNonP':
                                par.add('tau', init['tau'])
                                par.add('N', init['N'] * 1e6)
                                paras = [para['mr'], para['Eg']]
                        elif model == 'Line':
                                par.add('A', init['A'])
                                par.add('B', init['B'])
                                paras = [0]
                        else:
                                model = 'Drude'
                                par.add('tau', init['tau'])
                                par.add('N', init['N'] * 1e6)
                                paras = [para['mr']]
                                wn.warn('Warning:: Model undefined or' +
                                        ' not understood, Drude model ' +
                                        'chose as default', RuntimeWarning)
                guessed = mod.Switch(model, par, x, paras)
                if np.any(err):
                        data = np.append(data, err, axis=0)
                        data = np.reshape(data, (2, len(y)))
                res = ResWrap(model, paras, c)
                out = fit.minimize(res, par, args=(x,), kws={'data': data},
                                   nan_policy='omit')
                fitted = mod.Switch(model, out.params, x, paras)

                if plot:
                        print(fit.fit_report(out))
                        if c:
                                col = 2
                        elif not c:
                                col = 1
                        fitFig, axes = plt.subplots(nrows=1,
                                                    ncols=col,
                                                    sharex='all',
                                                    squeeze=True)

                        if c:
                                ebar = axes[0].errorbar(x,
                                                        np.real(y),
                                                        np.real(err),
                                                        marker='o', ls='')
                                c0 = ebar[0].get_color()
                                axes[0].plot(x, np.real(fitted),
                                             ls='-', marker='',
                                             color=c0)
                                if guess:
                                        axes[0].plot(x, np.real(guessed),
                                                     ls='--',
                                                     marker='', color=c0)
                                axes[1].errorbar(x, np.imag(y), np.imag(err),
                                                 marker='o', ls='', color=c0)
                                axes[1].plot(x, np.imag(fitted),
                                             ls='-', marker='', color=c0)
                                if guess:
                                        axes[1].plot(x, np.imag(guessed),
                                                     ls='--', marker='',
                                                     color=c0)
                                axes[0].set_xlabel('$\\nu$(THz)', x=1.05)
                                axes[0].set_ylabel(yLabel)
                                axes[1].yaxis.set_label_position('right')
                                axes[1].tick_params(which='major', right=True,
                                                    left=False)
                                axes[1].yaxis.set_ticks_position('right')

                        elif not c:
                                print(len(x), len(y))
                                ebar = axes.errorbar(x,
                                                     np.real(y),
                                                     np.real(err),
                                                     marker='o', ls='')
                                c0 = ebar[0].get_color()
                                axes.plot(x, np.real(fitted),
                                          ls='-', marker='',
                                          color=c0)
                                if guess:
                                        axes.plot(x, np.real(guessed),
                                                  ls='--',
                                                  marker='', color=c0)
                                axes.set_xlabel('$\\nu$(THz)')
                                axes.set_ylabel(yLabel)
                return out.params
