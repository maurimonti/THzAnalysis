import numpy as np  # No need to explain this
import sympy as sym  # symbolic calculations
import scipy.integrate as spint  # Integration tools
import sys  # system methods, used for exit


def zeropad(xData, yData, exp=0):
        xOut = []
        yOut = []
        if exp > 0:
                step = xData[2] - xData[1]
                xOut = xData
                yOut = yData
                x0 = xData[-1]

                for x in range(exp * len(xData)):
                        xOut = np.append(xOut, x0 + x * step)
                        yOut = np.append(yOut, 0.0)

        else:
                xOut = xData
                yOut = yData

        return xOut, yOut


def FFT(xData, yData, xUnit=0):
    # computes the FFT of the data and returns the frequency and the spectrum
    # of the FFT
    # xUnit used to convert the xData in frequency depending on the unit of
    # the x (mm or seconds, not implemented yet)
    yFourier = np.fft.fft(yData)
    fftLen = len(yFourier)
    yFourier = yFourier[0:int((fftLen / 2 + 1))]
    if xUnit == 'OD':
        conv = 0.2998
    else:
        conv = 0.1499

    timeStep = abs(xData[fftLen - 1] - xData[0]) / (fftLen - 1) / conv
    freq = np.array(list(range(int(fftLen / 2 + 1)))) / timeStep / fftLen

    return freq, yFourier


def IFFT(xData, yData, xUnit='mm'):
    # will do the inverse fft
    yIFourier = np.fft.ifft(yData)
    fftLen = len(yIFourier)
    yIFourierOut = yIFourier[0:int((fftLen / 2 + 1))]
    if xUnit == 'OD':
        conv = 0.2998
    elif xUnit == 'mm':
        conv = 0.1499
    elif xUnit == 'ps':
        conv = 1.0
    else:
        conv = 0.1499

    timeStep = abs(xData[fftLen - 1] - xData[0]) / (fftLen - 1) / conv
    freq = np.array(list(range(int(fftLen / 2 + 1)))) / timeStep / fftLen

    return freq, yIFourierOut


def SymHessian(f, x, y):
    # Computes and returns the symbolic hessian
    # of a scalar function of two variables
    # f: functions to differentiate
    # x, y variables to respect with differentiate
    H = [[[], []], [[], []]]
    H[0][0] = sym.diff(f, x, x)
    H[0][1] = sym.diff(f, x, y)
    H[1][0] = sym.diff(f, y, x)
    H[1][1] = sym.diff(f, y, y)

    return H


def SymGradient(f, x, y):
    # Computes and returns the symbolic gradient
    # of a scalar function of two variables
    # f: functions to differentiate
    # x, y variables to respect with differentiate
    D = [[], []]
    D[0] = sym.diff(f, x)
    D[1] = sym.diff(f, y)

    return D


def Gradient(x, step=0):
    xGrad = np.gradient(x)
    if step <= 0:
        step = np.abs(x[-1] - x[-2])
    gradient = xGrad / step

    return gradient


def Hessian(x, step1, step2):
    xGrad = np.gradient(x)
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype)
    for k, kgrad in enumerate(xGrad):
        tmp_grad = np.gradient(kgrad)
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl / (step1 * step2)
    return hessian


def AntiDerivative(x, y):
    out = np.zeros(len(x))
    for i in range(len(x)):
        tmp = 0
        tmp = spint.quad(y, 0, x[i])
        out[i] = tmp[0]
    return x, out


def WavenumberConverter(wn, out='nm'):

        if out is 'micro':
                xOut = [x**(-1) * 10000 for x in wn]

        elif out is 'nm':
                xOut = [x**(-1) * 10000000 for x in wn]

        elif out is 'eV':
                xOut = [x * 1.23984 * 0.0001 for x in wn]
        else:
                sys.exit('Damn! Unit not recognized!')

        return xOut


def SymConvolution(f, g, t, lower_limit=-sym.oo, upper_limit=sym.oo):
    tau = sym.Symbol('tau', real=True)
    CI = sym.integrate(f.subs(t, tau) * g.subs(t, t - tau),
                       (tau, lower_limit, upper_limit))
    return CI


def Function(model, para, x):
    if model == 'Gaussian' or model == 'Gauss' or model == 'gaussian':
        return(Gaussian(para[0], para[1], x))
    elif model == 'Lorentzian' or model == 'Lorentz' or model == 'lorentzian':
        return(Lorentzian(para[0], para[1], x))
    elif model == 'Logistic' or model == 'logistic' or model == 'sigmoid':
        return(Logistic(para[0], para[1], x))
    else:
        sys.exit('Function not recognized: check spelling and capitalization')


def Gaussian(mu, sigma, x):
    G = np.exp(-(x - mu)**2 / (2 * sigma**2))
    return G


def Lorentzian(x0, gamma, x):
    L = (gamma / (2 * np.pi)) / ((x - x0)**2 + (0.5 * gamma)**2)
    return L


def Logistic(x0, k, x):
    L = 1 / (1 + np.exp(-2 * k * (x - x0)))
    return L
