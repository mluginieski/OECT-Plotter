import numpy
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import warnings
import Settings as S
from PlotterClass import Plotter

def ExpDec(xData, yData, Vds, Vgs):

    # model for fitting
    def func(x, A, B, tau):
        return  A + B * numpy.exp(-x/tau)
    
    # function for genetic algorithm to minimize (sum of squared error)
    def sumOfSquaredError(parameterTuple):
        warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
        val = func(xData, *parameterTuple)
        return numpy.sum((yData - val) ** 2.0)

    def generate_Initial_Parameters():

        parameterBounds = []
        parameterBounds.append([0.0, 1e8])  # search bounds for A
        parameterBounds.append([0.0, 1e5])  # search bounds for B
        parameterBounds.append([0.0, 10])   # search bounds for tau
        
        # "seed" the numpy random number generator for repeatable results
        result = differential_evolution(sumOfSquaredError, parameterBounds, tol=1e-9, seed=3)
        return result.x

    # generate initial parameter values
    geneticParameters = generate_Initial_Parameters()

    # curve fit the test data
    testParameters, test_pcov = curve_fit(func, xData, yData, geneticParameters)
    test_standardError = numpy.sqrt(numpy.diag(test_pcov))

    p0 = [testParameters[0], testParameters[1], testParameters[2]]
    
    fittedParameters, pcov = curve_fit(func, xData, yData, p0)
    standardError = numpy.sqrt(numpy.diag(pcov))

    modelPredictions = func(xData, *fittedParameters) 

    absError = modelPredictions - yData

    SE = numpy.square(absError) # squared errors
    MSE = numpy.mean(SE) # mean squared errors
    RMSE = numpy.sqrt(MSE) # Root Mean Squared Error, RMSE
    Rsquared = 1.0 - (numpy.var(absError) / numpy.var(yData))
    #f.write('\nRMSE: {}\n'.format(RMSE))
    #f.write('R-squared: {}\n'.format(Rsquared))

    ##########################################################
    # graphics output section
    plotter = Plotter(S.timelbl, S.Igslbl, S.timeUn, S.mIUn)
    plotter.set_folder('Graphs/Transient/ExpDec')
    
    textstr = '\n'.join((r'$V_D$ = %.2f V' % (Vds),
                             r'$V_G$ = %.2f V' % (Vgs),
                             r'                           ',
                             r'$f(x) = A + B\cdot\exp(-x/\tau)$',
                             r'$A$ = %.2f $\pm$ %.3f $\mu -$' % (fittedParameters[0]/1e-6,standardError[0]/1e-6),
                             r'$B$ = %.2f $\pm$ %.3f $\mu -$'  % (fittedParameters[1]/1e-6,standardError[1]/1e-6),
                             r'$\tau$ = %.2f $\pm$ %.3f $ms$'   % (fittedParameters[2]/1e-3,standardError[2]/1e-3),
                             r'$r^2$ = %.2f' %(Rsquared)
                             ))
    
    plotter.add_plot(xData, yData/1e-3, plot_type='scatter', annotate_text=textstr, annotate_pos=(0.5,0.7), annotate_cords='axes fraction', label='Exp')

    # create data for the fitted equation plot
    xModel = numpy.linspace(min(xData), max(xData), 10000)
    yModel = func(xModel, *fittedParameters)
    
    plotter.add_plot(xModel, yModel/1e-3, color='red', label='Fit')

    plotter.plot(xlog_scale=True, save_name=f'Igxt_Fit_Vd{Vds}V_Vg{Vgs}V', close=True)

    return fittedParameters[2], standardError[2]

def FariaDuong(time, Ids, Igs, I0, Vds, Vgs):

    def IgFit(xData, yData):
        # model for fitting
        def funcIg(x, Rd, Rs, Cd):
            return  (Vgs/(Rd + Rs)) + ((Vgs * Rd)/(Rs*(Rd + Rs)) * numpy.exp(-(Rd + Rs)*x/(Cd * Rd * Rs)))

        # function for genetic algorithm to minimize (sum of squared error)
        def sumOfSquaredError(parameterTuple):
            warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
            val = funcIg(xData, *parameterTuple)
            return numpy.sum((yData - val) ** 2.0)

        def generate_Initial_Parameters():

            parameterBounds = []
            parameterBounds.append([0.0, 1e10])  # search bounds for Rd
            parameterBounds.append([0.0, 1e4])   # search bounds for Rs
            parameterBounds.append([0.0, 1e-2])  # search bounds for Cd
        
            # "seed" the numpy random number generator for repeatable results
            result = differential_evolution(sumOfSquaredError, parameterBounds, tol=1e-9, seed=3)
            return result.x

        # generate initial parameter values
        geneticParameters = generate_Initial_Parameters()

        # curve fit the test data
        testParameters, test_pcov = curve_fit(funcIg, xData, yData, geneticParameters)
        test_standardError = numpy.sqrt(numpy.diag(test_pcov))

        p0 = [testParameters[0], testParameters[1], testParameters[2]]
    
        fittedParameters, pcov = curve_fit(funcIg, xData, yData, p0)
        standardError = numpy.sqrt(numpy.diag(pcov))

        modelPredictions = funcIg(xData, *fittedParameters) 

        absError = modelPredictions - yData

        SE = numpy.square(absError) # squared errors
        MSE = numpy.mean(SE) # mean squared errors
        RMSE = numpy.sqrt(MSE) # Root Mean Squared Error, RMSE
        Rsquared = 1.0 - (numpy.var(absError) / numpy.var(yData))
        #f.write('\nRMSE: {}\n'.format(RMSE))
        #f.write('R-squared: {}\n'.format(Rsquared))

        ##########################################################
        # graphics output section
        plotter = Plotter(S.timelbl, S.Igslbl, S.timeUn, S.mIUn)
        plotter.set_folder('Graphs/Transient/FariaDuong/Igs')
    
        textstr = '\n'.join((r'$V_{DS}$ = %.2f V'                   % (Vds),
                             r'$V_{DS}$ = %.2f V'                   % (Vgs),
                             r'$R_{d}$ = %.2f $\pm$ %.3f $M\Omega$' % (fittedParameters[0]/1e9,standardError[0]/1e9),
                             r'$R_{s}$ = %.2f $\pm$ %.3f $k\Omega$' % (fittedParameters[1]/1e3,standardError[1]/1e3),
                             r'$C_{d}$ = %.2f $\pm$ %.3f $\mu F$'   % (fittedParameters[2]/1e-6,standardError[2]/1e-6),
                             r'$r^2$ = %.2f'                        % (Rsquared)
                             ))
    
        plotter.add_plot(xData, yData/1e-3, plot_type='scatter', annotate_text=textstr, annotate_pos=(0.5,0.7), annotate_cords='axes fraction', label='Exp')

        # create data for the fitted equation plot
        xModel = numpy.linspace(min(xData), max(xData), 10000)
        yModel = funcIg(xModel, *fittedParameters)
    
        plotter.add_plot(xModel, yModel/1e-3, color='red', label='Fit')

        plotter.plot(xlog_scale=True, save_name=f'Igxt_Fit_Vd{Vds}V_Vg{Vgs}V', close=True)

        return fittedParameters, standardError
    
    def IdFit(xData, yData, Param):

        Rd,Rs,Cd,gm = Param

        # model for fitting
        def funcId(x, f):
            return I0 + ((Vgs*(gm*Rd - f))/(Rd+Rs)) - (((Vgs*Rd*(gm*Rs + f))/(Rs*(Rd+Rs))) * numpy.exp(- ((Rd + Rs) * x) / (Cd * Rd * Rs)))

        # function for genetic algorithm to minimize (sum of squared error)
        def sumOfSquaredError(parameterTuple):
            warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
            val = funcId(xData, *parameterTuple)
            return numpy.sum((yData - val) ** 2.0)

        def generate_Initial_Parameters():

            parameterBounds = []
            parameterBounds.append([0.0, 1])  # search bounds for f
           
            # "seed" the numpy random number generator for repeatable results
            result = differential_evolution(sumOfSquaredError, parameterBounds, tol=1e-9, seed=3)
            return result.x

        # generate initial parameter values
        geneticParameters = generate_Initial_Parameters()

        # curve fit the test data
        testParameters, test_pcov = curve_fit(funcId, xData, yData, geneticParameters)
        test_standardError = numpy.sqrt(numpy.diag(test_pcov))

        p0 = testParameters[0]
    
        fittedParameters, pcov = curve_fit(funcId, xData, yData, p0)
        standardError = numpy.sqrt(numpy.diag(pcov))

        modelPredictions = funcId(xData, *fittedParameters) 

        absError = modelPredictions - yData

        SE = numpy.square(absError) # squared errors
        MSE = numpy.mean(SE) # mean squared errors
        RMSE = numpy.sqrt(MSE) # Root Mean Squared Error, RMSE
        Rsquared = 1.0 - (numpy.var(absError) / numpy.var(yData))
        #f.write('\nRMSE: {}\n'.format(RMSE))
        #f.write('R-squared: {}\n'.format(Rsquared))

        ##########################################################
        # graphics output section
        plotter = Plotter(S.timelbl, S.Idslbl, S.timeUn, S.mIUn)
        plotter.set_folder('Graphs/Transient/FariaDuong/Ids')
    
        textstr = '\n'.join((r'$V_{DS}$ = %.2f V'        % (Vds),
                             r'$V_{DS}$ = %.2f V'        % (Vgs),
                             r'$f$   = %.2f $\pm$ %.3f ' % (fittedParameters[0],standardError[0]),
                             r'$I_0$ = %.2e A'           % (I0),
                             r'$g_m$ = %.2e S'           % (gm),
                             r'$R_d$ = %.2f $k\Omega$'   % (Rd/1000),
                             r'$R_s$ = %.2f $\Omega$'    % (Rs),
                             r'$C_d$ = %.2f $\mu F$'     % (Cd/1e-6),
                             r'$r^2$ = %.2f'             % (Rsquared)
                             ))
    
        plotter.add_plot(xData, yData/1e-3, plot_type='scatter', annotate_text=textstr, annotate_pos=(0.5,0.7), annotate_cords='axes fraction', label='Exp')

        # create data for the fitted equation plot
        xModel = numpy.linspace(min(xData), max(xData), 10000)
        yModel = funcId(xModel, *fittedParameters)
    
        plotter.add_plot(xModel, yModel/1e-3, color='red', label='Fit')

        plotter.plot(xlog_scale=True, save_name=f'Idxt_Fit_Vd{Vds}V_Vg{Vgs}V', close=True)

        return fittedParameters, standardError

    fit_Par, std_Err = IgFit(time, Igs)

    IdsOn = numpy.mean(Ids[len(Ids)-5000:])
    
    gm = abs(IdsOn - I0) / Vgs
    #gm = (I0 - IdsOn) / Vgs

    Param = [fit_Par[0],fit_Par[1],fit_Par[2],gm]

    fit_Par, std_Err = IdFit(time, Ids, Param)

    Param.append(fit_Par[0])

    return Param
