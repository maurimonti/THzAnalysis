import sys  # system methods, used for exit
import numpy as np  # Do I really have to explain that?
import matplotlib.pyplot as plt  # plotting
import scipy.signal as sig  # library with signal analysis (find peaks, et.)

import lmfit as fit  # fitting package

import src.math_functions as mt  # set of mathematical tools

# ci starebbe bene un affare per eliminare l'offset
# tipo che prende un po' di dati all'inizio(fine), tipo 10 punti, controlla che
# non siano troppo alti e fa la media e la usa come offset


def Window(model, para, x, y):
    wd = mt.Function(model, para, x)
    yout = y * wd
    return x, yout


def Plotter(par, x, func, xFlag=None, fit=None):
    tau = par[:, 0]
    tauErr = par[:, 1]
    if fit:
        a = fit[0]
        b = fit[1]
        aErr = fit[2]
        # bErr = fit[3]

    fig = plt.figure()

    if func == 'Drude':
        N = par[:, 2]
        NErr = par[:, 3]

        ax1 = plt.subplot(211)
        ax1.errorbar(x, tau * 1e15, tauErr * 1e15,
                     marker='s', ls='', c='b')
        plt.ylabel('$\\tau(fs)$')
        plt.axis([x[0] - 0.01 * x[-1], x[-1] + 0.01 * x[-1],
                 np.min(tau) - np.max(tauErr),
                 np.max(tau) + np.max(tauErr)])
        ax1.set_xticklabels([])

        ax2 = plt.subplot(212)
        ax2.errorbar(x, N, NErr,
                     marker='d', ls='', c='r')
        plt.axis([x[0] - 0.01 * x[-1], x[-1] + 0.01 * x[-1],
                 np.min(N) - np.max(NErr),
                 np.max(N) + np.max(NErr)])
        plt.ylabel('$N(cm^{-3})$')
        plt.xlabel('$t_p(ps)$')

    if func == 'Cyclotron':
        # A = par[:, 2]
        # AErr = par[:, 3]
        fqC = par[:, 4]
        fqCErr = par[:, 5]
        gamma = np.sqrt(2 * fqC * 2e12 /
                        np.abs(tau)) / (np.pi * 2e12)
        gammaErr = tauErr * gamma / tau

        ax1 = plt.subplot(211)
        ax1.errorbar(x, fqC, fqCErr,
                     fmt='s', c='b', label='exp. points')
        if fit:
            ax1.plot(x, a + b * x, ls='-', c='r', label='Fit')
            plt.text(0.5, 0.5, '$\\nu_C$ = B*' +
                     str(np.round(b,
                                  decimals=3)) + ' + ' +
                     str(np.round(a, 3)))
            plt.text(0.5, 0.2, 'm/m$_0$ = ' +
                     str(np.round(1.6e-19 / (2e12 * np.pi * b * 9.11e-31),
                                  decimals=5)) +
                     ' +/- ' +
                     str(np.round(1.6e-19 * aErr /
                                  (2e12 * np.pi * 9.11e-31 * b**2),
                                  decimals=5)))
        plt.ylabel('$\\nu_C$(THz)')
        ax1.set_xticklabels([])
        plt.axis([x[0] - 0.05, x[-1] + 0.05,
                 np.min(fqC) - np.max(fqCErr),
                 np.max(fqC) + np.max(fqCErr)])
        if xFlag:
            plt.axis([x[0] - 0.1 * x[-1], x[-1] + 0.1 * x[-1],
                     np.min(fqC) - np.max(fqCErr),
                     np.max(fqC) + np.max(fqCErr)])
        if fit:
            plt.legend(loc='best')

        ax2 = plt.subplot(212)
        ax2.errorbar(x, gamma,
                     gammaErr,
                     marker='d', ls='-', c='r')
        plt.ylabel('$\Gamma$(THz)')
        plt.xlabel('B(T)')
        plt.axis([x[0] - 0.05, x[-1] + 0.05,
                 np.min(gamma) - np.max(gammaErr),
                 np.max(gamma) + np.max(gammaErr)])
        if xFlag:
            plt.axis([x[0] - 0.1 * x[-1], x[-1] + 0.1 * x[-1],
                     np.min(gamma) - np.max(gammaErr),
                     np.max(gamma) + np.max(gammaErr)])
            plt.xlabel('t(ps)')
        return fig


def Extrema(fMin, fMax, f, I):
    idxMin = np.where(np.abs(f - fMin) < 0.3)
    idxMin = idxMin[0]
    idxMin = idxMin[-1]
    idxMax = np.where(np.abs(f - fMax) < 0.3)
    idxMax = idxMax[0]
    idxMax = idxMax[0]
    IMin = np.min(I[:, idxMin:idxMax])
    IMax = np.max(I[:, idxMin:idxMax])
    return idxMin, idxMax, IMin, IMax


def Interpolation(x1, y1, x2, y2, zero1=0, zero2=0):
    step = x1[2] - x1[1]
    x1 = x1 - zero1
    x2 = x2 - zero2
    # print(x1[0], x2[0], x1[-1], x2[-1])
    gap1 = x1[0] - x2[0]
    gap2 = x1[-1] - x2[-1]
    # print(gap1, gap2, step)
    x01 = x1[-1]
    x02 = x2[-1]
    # plt.plot(x1, y1)
    # plt.plot(x2, y2)
    # plt.show()
    if gap1 < 0:
        xtmp1 = x1
        ytmp1 = y1
        xtmp2 = []
        ytmp2 = []
        for x in range(np.int(np.abs(gap1 / step))):
            xtmp2 = np.append(xtmp2, x * step)
            ytmp2 = np.append(ytmp2, 0.0)
        xtmp2 = np.append(xtmp2, + x2)
        ytmp2 = np.append(ytmp2, y2)
    elif gap1 == 0:
        xtmp1 = x1
        ytmp1 = y1
        xtmp2 = x2
        ytmp2 = y2
    elif gap1 > 0:
        xtmp1 = []
        ytmp1 = []
        xtmp2 = x2
        ytmp2 = y2
        for x in range(np.int(np.abs(gap1 / step))):
            xtmp1 = np.append(xtmp1, x * step)
            # print(xtmp1[-1])
            ytmp1 = np.append(ytmp1, 0.0)
        # print(xtmp1)
        # plt.plot(xtmp1, ytmp1)
        # plt.show()
        xtmp1 = np.append(xtmp1, x1)
        # print(xtmp1)
        ytmp1 = np.append(ytmp1, y1)
        # print(len(xtmp1), len(ytmp1))
        # plt.plot(xtmp1, ytmp1)
        # # plt.plot(xtmp2, ytmp2)
        # plt.show()

    if gap2 > 0:
        xtmptmp1 = xtmp1
        ytmptmp1 = ytmp1
        xtmptmp2 = xtmp2
        ytmptmp2 = ytmp2
        # print(xtmptmp2)
        for x in range(np.int(np.abs(gap2 / step))):
            xtmptmp2 = np.append(xtmptmp2, x02 + x * step)
            ytmptmp2 = np.append(ytmptmp2, 0.0)
        # print(ytmptmp1)
    elif gap2 == 0:
        xtmptmp1 = xtmp1
        ytmptmp1 = ytmp1
        xtmptmp2 = xtmp2
        ytmptmp2 = ytmp2
    elif gap2 < 0:
        xtmptmp1 = xtmp1
        ytmptmp1 = ytmp1
        xtmptmp2 = xtmp2
        ytmptmp2 = ytmp2
        for x in range(np.int(np.abs(gap2 / step))):
            xtmptmp1 = np.append(xtmptmp1, x01 + x * step)
            ytmptmp1 = np.append(ytmptmp1, 0.0)

    xOut1 = xtmptmp1 + zero1
    yOut1 = ytmptmp1
    xOut2 = xtmptmp2 + zero2
    yOut2 = ytmptmp2

    return xOut1, yOut1, xOut2, yOut2


def Flipper(y):
    # Stupid wrapper that flips an 1D array
    # y: data array to flip, 1D array
    # returns yOut the flipped array
    yOut = np.flip(y)

    return yOut


def Reader(path, comment='%', delimiter='\t',
           skipRows=0, skipEnd=0, usecols=(), caller='',
           TW=False, transpose=True):
    # Just a wrapper to read text files
    # comment: comments used in the file, default %, string
    # delimiter: the delimiter used in the file, default tab (\t), string
    # skipRows: number of rows to skip at the beginning of a file,
    # default to 0, int
    # returns an array with the red data, unpacked
    try:
        data = np.genfromtxt(path,
                             comments=comment,
                             delimiter=delimiter,
                             skip_header=skipRows,
                             skip_footer=skipEnd,
                             unpack=False,
                             usecols=usecols, )

        datanew = data[:, ~np.all(np.isnan(data), axis=0)]
        data = 0
        if transpose:
            data = np.transpose(datanew)
        else:
            data = datanew
        # print(np.transpose(datanew))
        # data = data[:, ]
        # data = np.loadtxt(path,
        #                   comments=comment,
        #                   delimiter=delimiter,
        #                   skiprows=skipRows,
        #                   unpack=True,
        #                   usecols=usecols)
    except IOError:
        sys.exit('Damn! File ' + path + ' not found by ' + caller + '!')

    return data


def Shifter(xData, xDataRef):
    # Aligns the pulses, returns the aligned x axes
    # xData: first pulse x axis, array
    # xDataRef: reference pulse x axis, array
    # returns: xOut, xOutRef the shifted x axes of the pulse and reference,
    # arrays
    delta = xData[0] - xDataRef[0]
    xOut = xData - delta
    xOutRef = xDataRef

    return xOut, xOutRef


def Computer(data, dataRef, dataBack=[], dataBackRef=[], Back=False):
    # This function computes the reflectivity spectrum dividing the raw data
    # for a given reference. If Back is True also subtract a given background
    # to compensate for a non good alignment or for a non good aperture size.
    # data: the signal we are interested in, array
    # dataRef; the reference signal, array
    # dataBack: the background data for our signal, default empty list, list
    # dataBackRef: background data for the reference, default empty, list
    # Back: if True enables the background subtraction, default False,
    # boolean value
    # Returns the ratio between the signal and the reference, array

    if Back is True:
        spectrum = data - dataBack
        spectrumRef = dataRef - dataBackRef

    elif Back is False:
        spectrum = data
        spectrumRef = dataRef

    ratio = spectrum / spectrumRef
    return ratio


def plotter(x, y, shape, name, flag='normal'):
    # This function plots the data,
    # x: the x value to plot, list or array,
    # y: the y value to plot, list or array
    # shape: the symbols shape and colour for the plot, string
    # name: the legend for the dataset, string
    # flag: the typology of plot:
    #       normal: standard linear plot
    #       logx: logarithmic x axis
    #       logy: logarithmic y axis
    #       loglog: both axes on logarithmic scale
    # default to normal, string
    # Returns outPlot, a plot object

    if flag is 'normal':
        outPlot = plt.plot(x, y, shape, label=name)
    elif flag is 'logx':
        outPlot = plt.semilogx(x, y, shape, label=name)
    elif flag is 'logy':
        outPlot = plt.semilogy(x, y, shape, label=name)
    elif flag is 'loglog':
        outPlot = plt.loglog(x, y, shape, label=name)
    else:
        sys.exit('damn! plot type non recognized!')

    return outPlot


def Normalizer(data, default=0, noisy=False, theoMax=101):
    # Normalization of the data, to a default value called default. If 0 the
    # data will be divided for the maximum. Another parameter noisy can be
    # called if the data are not smooth and deletes all the values that are
    # above a theoretical maximum theoMax.
    # data: the data to normalise, array
    # default: the value to which normalise the data, if 0 the data will be
    # normalised for the maximum, default 0, float
    # noisy: parameter that triggers the clean of the data, if some points are
    # above a certain theoretical value theoMax, those won't be counted as
    # maximum, default False, boolean
    # theoMax: maximum theoretical value for the noisy feature,
    # default 101, float
    # returns out, the normalised data, array

    if default is 0:
        if noisy is False:
            maxVal = np.nanmax(data)
            out = data / maxVal
            return out

        elif noisy is True:
            maximaVal = data[sig.argrelmax(data)]
            maximaVal = maximaVal[np.where(maximaVal < theoMax)]  # deletes
            maxVal = np.nanmax(maximaVal)                         # spikes
            out = data / maxVal
            return out

        else:
            sys.exit('WTF? I don\'t know how the noise is!')

    elif default < 0:
        sys.exit('Damn! Normalization value is negative!')

    else:
        out = data / default
        return out


def Converter(xIn, yIn=None, xFormat='', yFormat='', xUnit='', yUnit=''):
    # This function converts the x in a chosen unit that depends on the xUnit
    # value.
    # xIn: x data to be converted
    # yIn: y data to be converted, default None, list
    # xFormat: unit of input data, default '', string
    # accepted keys: 'cm1' ( cm^-1), 'mm' (millimiters)
    # yFormat: unit of input y data, default '', string
    # accepted keys: 'ratio' (pure number)
    # xUnit: unit to which convert the x data, default '', string
    # accepted keys for xFormat = 'cm1': 'cm1' (cm^-1), 'micro' (micrometers),
    # 'nm' (nanometers), 'eV' (electronVolt)
    # accepted keys for xFormat = 'mm': 'THZ' (terahertz), 'ps' (picoseconds)
    # yUnit: unit to which convert the y data, default '', string
    # accepted keys for yFormat = 'ratio': '%' (percentage)
    # Returns out which contains the converted data, array
    if xFormat is '' or xUnit is '':
        xOut = xIn

    elif xFormat is 'cm1':

        if xUnit is not 'cm1':
            xOut = mt.WavenumberConverter(xIn, xUnit)
        elif xUnit is 'cm1':
            xOut = [x for x in xIn]

    elif xFormat is 'mm':

        if xUnit is 'THz':
            xOut = xIn
        elif xUnit is 'ps':
            xOut = [x / 0.15 for x in xIn]
        else:
            sys.exit('damn! Unit of measure of x not recognized!')
    if yIn is not None:
        if yFormat is '' or yUnit is '':
                yOut = yIn

        elif yFormat is 'ratio':
            if yUnit is '%':
                yOut = [y * 100 for y in yIn]
        out = np.array([xOut, yOut])
    elif yIn is None:
        out = np.array(xOut)

    return out


def FindPeaks(freq, spectrum, widths, Top=101):
    # Uses find_peaks_cwt of the Scipy.signal library to find peaks, then
    # returns the peak positions and heights. Top is needed to delete all the
    # Nonphysical peaks that are above the maximum possible value, not
    # efficient yet
    # freq: input frequency/x data of spectrum, array
    # spectrum: input signal/y data, array
    # widths: widths to use in the find_peaks_cwt function, array
    # Top: maximum value expected, used to clean the spectrum from the
    # spikes, not fully working, default 101, float
    # returns peaksPos: position of the peaks, array and
    # peaksHeigth: height of the peaks found, array

    # delete spikes better

    cleanedFreq = freq[np.where(spectrum < Top)]
    cleanedSpectrum = spectrum[np.where(spectrum < Top)]

    widths = widths / (freq[1] - freq[0])

    idx = sig.find_peaks_cwt(cleanedSpectrum, widths)

    peaksPos = cleanedFreq[idx]
    peaksHeight = cleanedSpectrum[idx]

    return peaksPos, peaksHeight


def LabelAssigner(xUnit, yUnit='', meas='reflection'):
    # Self-explaining receives the unit chosen and returns the correct label
    # xUnit: unit of the input x data, string
    # accepted keys: 'cm1' (cm^-1), 'nm' (nanometers), 'micro' (micrometers),
    # 'eV' (electronvolt), 'THz' (terahertz)
    # yUnit: unit of the input y data, default to '', string
    # accepted values for meas='reflections', '%' (percentage)
    # accepted values for meas='transmission', '%' (percentage)
    # meas: type of measurement we are performing, default 'reflection', string
    # accepted keys: 'reflection', 'transmission'
    # returns the x and y labels to use in a plot, string

    # add case sensitive stuff on reflection-transmission

    if xUnit is 'cm1':
        xlabel = '$k(cm^{-1})$'
    elif xUnit is 'nm':
        xlabel = '$\lambda(nm)$'
    elif xUnit is 'micro':
        xlabel = '$\lambda(\mu m)$'
    elif xUnit is 'eV':
        xlabel = '$E(eV)$'
    elif xUnit is 'THz':
        xlabel = '$\\nu(THz)$'
    if meas is 'reflection':
        if yUnit is '%':
            ylabel = '$r(\%)$'
        elif yUnit is '':
            ylabel = '$r$'
    elif meas is 'transmission':
        if yUnit is '%':
            ylabel = '$t(\%)$'
        elif yUnit is '':
            ylabel = '$t$'

    return xlabel, ylabel


def FitPeaks(freq, spectrum, shape, peakPos, width):
        # Tries to fit the peaks found, requires the position and width
        # to be given. Uses the lmfit package
        # freq: x axis, array
        # spectrum: y axis, array
        # shape: expected shape of the function, only Gaussian and
        # Lorentzian implemented, string
        # peakPos: array with the peak positions, array
        # width: expected widths of the peaks, array
        # returns the fit parameters of the peaks, array
        N = 3
        if shape is 'gaussian':
                mod = fit.models.GaussianModel()
        elif shape is 'lorentzian':
                mod = fit.model.lorentzian()
        else:
                sys.exit('Damn! Model not found!')

        # extract peaks
        step = freq[1] - freq[0]
        numPeaks = len(peakPos)
        dataPeaks = np.zeros((numPeaks, 2,
                             np.int(
                              np.floor(
                                  2 * N * np.amax(width) / step))))

        for i in range(numPeaks - 1):
                last = np.int(np.floor(2 * N * width[i] / step))

                dataPeaks[i, 0, :last] = freq[
                    np.where(
                        np.abs(freq - peakPos[i]) < N * width[i])]
                dataPeaks[i, 1, :last] = spectrum[
                    np.where(
                        np.abs(freq - peakPos[i]) < N * width[i])]

        # fit peaks
        outPars = np.zeros((numPeaks, 2))
        for i in range(numPeaks - 1):
                pars = mod.guess(dataPeaks[i][1], x=dataPeaks[i][0])
                out = mod.fit(dataPeaks[i][1], pars, x=dataPeaks[i][0])
                p = out.params.valuesdict()
                outPars[i][0] = p['sigma']
                outPars[i][1] = p['center']
        return outPars

