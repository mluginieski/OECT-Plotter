import os
import numpy as np
import pandas as pd
import Settings as S
import ExpoModel as Exp
from PlotterClass import Plotter
from scipy.signal import find_peaks
from statistics import mean
from LinearModel import find_linear_region
from scipy.integrate import simpson

################ IxV ################
def IxV(fileNames):
    ''' Plots IxV curves in linear and log scale. Prints the resistance with the standard deviation '''
    ''' This function is written considering a dual sweep transfer curves '''
    
    plotter = Plotter(S.Voltlbl, S.Currlbl, S.VUn, S.IUn)
    plotter.set_folder('Graphs/IxV')
    
    for file in fileNames:
    
        df = pd.read_csv(file, skiprows=3, sep='\t')  ### Change accordingly to your data file
        devi, batc, Meas = fileNames[file]
        V = df.iloc[:, 3]                             ### Change accordingly to your data file
        I = df.iloc[:, 4]                             ### Change accordingly to your data file
        R = mean(V/I)
        Rstd = np.std(V/I)
        
        textstr = r'$R_{ch,0} = $ %.1f $\pm$ %.1f $M\Omega$' % (R/1e6, Rstd/1e6)
        plotter.add_plot(V, I, annotate_text=textstr, annotate_pos=(0.5,0.1), annotate_cords='axes fraction')
        plotter.plot(save_name=f'IxVLin_{devi}_{batc}_{Meas}', close=True)

        plotter.add_plot(V, abs(I))
        plotter.plot(ylog_scale=True, save_name=f'IxVLog_{devi}_{batc}_{Meas}', close=True)

################ Output ################
def Output(VGS, fileNames):
    ''' Plots output curves  '''
    
    plotter = Plotter(S.Vdslbl, S.Idslbl, S.VUn, S.mIUn)
    plotter.set_folder('Graphs/Output')
    
    for file in fileNames:

        df = pd.read_csv(file, skiprows=3, sep='\t')                     ### Change accordingly to your data file
        devi, batc, Meas = fileNames[file]
        
        if Meas == 'M1':
            MaxIds = max(df.iloc[:, 6])/1e-3                             ### Change accordingly to your data file
            MinIds = min(df.iloc[:, 6])/1e-3                             ### Change accordingly to your data file
            delta = abs(MaxIds * S.delta_scale)

        for Vgs in VGS:
            Vds = df[(round(df.iloc[:, 3], 1) == Vgs)].iloc[:, 5]        ### Change accordingly to your data file
            Ids = df[(round(df.iloc[:, 3], 1) == Vgs)].iloc[:, 6]        ### Change accordingly to your data file
            plotter.add_plot(Vds, Ids/1e-3, label=f'{Vgs} V')

        ylim = [MinIds - delta, MaxIds + delta]
        plotter.plot(legend_title=S.Vgslbl+' =', ylim=ylim, save_name=f'Output_{devi}_{batc}_{Meas}', close=True)

################ Transf ################
def Transf(VDS, fileNames):
    ''' Plots Transfer curves in linear and log scale, transconductance (gm), gm,max and onoff ratio '''
    ''' This function is written considering a dual sweep transfer curves '''

    aux = []
    for file in fileNames:
        df = pd.read_csv(file, skiprows=3, sep='\t')                                      ### Change accordingly to your data file
        devi, batc, Meas = fileNames[file]

        TransfDict = {'Vgs': [], 'Vds': [], 'Ids': [], 'Igs': []}
        for Vds in VDS:
            Ids = df[(round(df.iloc[:, 3], 1) == Vds)].iloc[:, 4].reset_index(drop=True)  ### Change accordingly to your data file
            Vgs = df[(round(df.iloc[:, 3], 1) == Vds)].iloc[:, 5].reset_index(drop=True)  ### Change accordingly to your data file
            Igs = df[(round(df.iloc[:, 3], 1) == Vds)].iloc[:, 6].reset_index(drop=True)  ### Change accordingly to your data file
            
            TransfDict['Vgs'].append(Vgs)
            TransfDict['Vds'].append(Vds)
            TransfDict['Ids'].append(Ids)
            TransfDict['Igs'].append(Igs)

            OnOff = Ids[int(len(Vgs)/2)] / Ids[0] ## Set for accumulation mode OECTs
                                                  ## For depletion mode OECTs, please change to: Ids[2] / Ids[int(len(Vgs)/2)]
            #### Transf - Linear ####
            plotter = Plotter(S.Vgslbl, S.Idslbl, S.VUn, S.mIUn)
            plotter.set_folder('Graphs/TransfLin')
    
            plotter.add_plot(Vgs, Ids/1e-3, label=S.Idslbl)
            plotter.add_plot(Vgs, Igs/1e-3, label=S.Igslbl)
            plotter.plot(legend_title=S.Vdslbl+f' = {Vds} V', save_name=f'TransfLin_{devi}_{batc}_{Meas}_Vds{round(Vds,1)}', close=True)

            #### Transf - Log ####
            plotter = Plotter(S.Vgslbl, S.Currlbl, S.VUn, S.IUn)
            plotter.set_folder('Graphs/TransfLog')
    
            plotter.add_plot(Vgs, abs(Ids), label=S.Idslbl)
            plotter.add_plot(Vgs, abs(Igs), label=S.Igslbl)
            plotter.plot(ylog_scale=True, legend_title=S.Vdslbl+f' = {Vds} V', save_name=f'TransfLog_{devi}_{batc}_{Meas}_Vds{round(Vds,1)}', close=True)

            # Transconductance
            plotter = Plotter(S.Vgslbl, S.gmlbl, S.VUn, S.mCondUn)
            plotter.set_folder('Graphs/Transc')
            
            dx = np.gradient(Vgs)
            dy = np.gradient(Ids)
            gm = dy/dx

            peaks, _ = find_peaks(abs(gm[:int(len(Vgs)/2)]))
            max_peak_idx = int(np.where(abs(gm) == max(abs(gm[peaks])))[0]) if len(peaks) > 0 else int(len(Vgs) / 2) - 1
            peak = str(np.round(abs(gm[max_peak_idx]/1e-3), decimals=1)).replace('[', '').replace(']', '')

            plotter.add_plot(Vgs[:int(len(Vgs)/2)], gm[:int(len(Vgs)/2)]/1e-3, annotate_text=peak, annotate_pos=(Vgs[max_peak_idx], abs(gm[max_peak_idx]/1e-3)), label='%.1f V' % (Vds))
            plotter.plot(legend_title=S.Vdslbl+' =', save_name=f'Transcond_{devi}_{batc}_{Meas}_Vds{round(Vds,1)}', close=True)

            aux.append([devi, batc, Meas, Vds, gm[max_peak_idx], OnOff])

        # Additional Graphs
        plotter = Plotter(S.Vgslbl, S.Idslbl, S.VUn, S.mIUn)
        plotter.set_folder('Graphs/TransfLin')
            
        if Meas == 'M1':
            MaxIds = max(df.iloc[:, 4])/1e-3
            MinIds = min(df.iloc[:, 4])/1e-3
            delta = abs(MaxIds * S.delta_scale)
        
        for i, vds in enumerate(TransfDict['Vds']):
            plotter.add_plot(TransfDict['Vgs'][i], TransfDict['Ids'][i]/1e-3, label=f'{vds} V')
        
        ylim = [MinIds - delta, MaxIds + delta]
        plotter.plot(legend_title=S.Vdslbl+' =', ylim=ylim, save_name=f'TransfLin_{devi}_{batc}_{Meas}_ALL-Vds', close=True)

        dfTransf = pd.DataFrame(aux, columns=['device', 'batch', 'Meas', 'Vds', 'gmmax', 'OnOff'])
        
        Vds   = dfTransf[(dfTransf.iloc[:,2] == Meas) & (dfTransf.iloc[:,0] == devi)].iloc[:,3].reset_index(drop=True)    
        gmmax = dfTransf[(dfTransf.iloc[:,2] == Meas) & (dfTransf.iloc[:,0] == devi)].iloc[:,4].reset_index(drop=True)
        OnOff = dfTransf[(dfTransf.iloc[:,2] == Meas) & (dfTransf.iloc[:,0] == devi)].iloc[:,5].reset_index(drop=True)

        # gm max
        plotter = Plotter(S.Vdslbl, S.gmmlbl, S.VUn, S.mCondUn)
        plotter.set_folder('Graphs/Param')
        
        plotter.add_plot(Vds, gmmax/1e-3, plot_type='scatter')
        plotter.plot(save_name=f'gmmax_{devi}_{batc}_{Meas}', close=True)
        
        # On/Off ratio
        plotter = Plotter(S.Vdslbl, S.ratlbl, S.VUn, S.ratUn)
        plotter.set_folder('Graphs/Param')

        plotter.add_plot(Vds, OnOff, plot_type='scatter')
        plotter.plot(save_name=f'OnOff_{devi}_{batc}_{Meas}', close=True)
        
    # Save Data
    f = 0
    fileOut = 'Outfile_Transf-{}.csv'.format(f)
    while os.path.isfile(fileOut) == True:
        fileOut = 'Outfile_Transf-{}.csv'.format(f)
        f += 1

    dfTransf.to_csv(fileOut, index=False)
    
################ VTC and Gain ################
def VTC(fileNames):
    ''' Plots a set of VTC and gain curves for different Vdd and a given load resistance '''
    ''' This function is written considering a dual sweep transfer curves '''
    
    VoutDict = {}
    
    for file, file_info in fileNames.items():
        
        df = pd.read_csv(file, skiprows=3, sep='\t')                            ### Change accordingly to your data file

        Vin  = df.iloc[:, 4].reset_index(drop=True)                             ### Change accordingly to your data file
        Vout = df.iloc[:, 2].reset_index(drop=True)                             ### Change accordingly to your data file
        devi = file_info[0]
        Vdd  = file_info[1]
        R    = file_info[2]

        if R not in VoutDict:
            VoutDict[R] = {'Vdd': [], 'Vin': [], 'Vout': []}
        
        VoutDict[R]['Vdd'].append(Vdd)
        VoutDict[R]['Vin'].append(Vin)
        VoutDict[R]['Vout'].append(Vout)

    ### VTC ###
    # Plot VTC for each resistance with varying Vdd
    plotter = Plotter(S.Vinlbl, S.Voutlbl, S.VUn, S.VUn)
    plotter.set_folder('Graphs/VTC')
        
    for R, vtc_data in VoutDict.items():
        for i, vdd in enumerate(vtc_data['Vdd']):
            plotter.add_plot(vtc_data['Vin'][i], vtc_data['Vout'][i], label=f'{vdd} V')

        if R/1e3 >= 1.0:
            fig_name = f'VTC_{devi}_R{round(R/1e3)}k'
        else:
            fig_name = f'VTC_{devi}_R{R}'
        plotter.plot(legend_title=S.Vddlbl+' =', save_name=fig_name, close=True)

    ### Gain ###
    # Plot Gain for each resistance with varying Vdd
    plotter = Plotter(S.Vinlbl, S.gnlbl, S.VUn, S.GnUn)
    plotter.set_folder('Graphs/VTC')
    for R, vtc_data in VoutDict.items():
        for i, vdd in enumerate(vtc_data['Vdd']):
            
            Vout  = vtc_data['Vout'][i]
            Vin   = vtc_data['Vin'][i]
            gnLen = int(len(Vin)/2)

            dx = np.gradient(Vin)
            dy = np.gradient(Vout)
            gn = dy/dx

            peaks, _ = find_peaks(abs(gn[:gnLen]))
            max_peak_idx = int(np.where(abs(gn[:gnLen]) == max(abs(gn[peaks])))[0])
            peak = str(np.round(abs(gn[max_peak_idx]), decimals = 1)).replace('[', '').replace(']', '')

            plotter.add_plot(Vin[:gnLen], abs(gn[:gnLen]), annotate_text=peak, annotate_pos=(Vin[max_peak_idx], abs(gn[max_peak_idx])), label=f'{vdd} V')
            
        if R/1e3 >= 1.0:
            fig_name = f'Gain_{devi}_R{round(R/1e3)}k'
        else:
            fig_name = f'Gain_{devi}_R{R}'
           
        plotter.plot(legend_title=S.Vddlbl+' =', save_name=fig_name, close=True)

################ OECT parameters trend ################            
def TrendPar(VDS, W, fileNames):
    ''' Plots OECT parameters as a function of the channel width '''
    
    for file in fileNames:

        df = pd.read_csv(file)
            
        gmmax = df[(df.iloc[:,2] == 'M3') & (df.iloc[:,3] == VDS)].iloc[:,4]
        onoff = df[(df.iloc[:,2] == 'M3') & (df.iloc[:,3] == VDS)].iloc[:,5]
    
        plotter = Plotter(S.Widlbl, S.gmlbl, S.ulenUn, S.mCondUn)
        plotter.set_folder('Graphs/Trend')

        plotter.add_plot(W/1e-6, gmmax/1e-3, plot_type='scatter', label=S.Vdslbl+f' = {VDS} V')
        plotter.plot(save_name=f'Trend_gmmax_Vds{VDS}', close=True)

        plotter = Plotter(S.Widlbl, S.ratlbl, S.ulenUn, S.ratUn)
        plotter.set_folder('Graphs/Trend')

        plotter.add_plot(W/1e-6, onoff, plot_type='scatter', label=S.Vdslbl+f' = {VDS} V')
        plotter.plot(save_name=f'Trend_onoff_Vds{VDS}', close=True)

################ Transfer characteristics trend ################
def TrendTransf(VDS, W, fileNames):
    ''' Plots transfer curves in linear and log scale and the transfer curves in respect to the channel width '''
    
    TransfDict = {'Vgs': [], 'devi' : [], 'Ids': [], 'Igs' : [], 'gm' : []}
    
    for file in fileNames:

        df = pd.read_csv(file, skiprows=3, sep='\t')                                    ### Change accordingly to your data file

        devi, batc, Meas = fileNames[file] # device

        if Meas == 'M3':
            Ids = df[(round(df.iloc[:,3],1) == VDS)].iloc[:,4].reset_index(drop=True)   ### Change accordingly to your data file
            Vgs = df[(round(df.iloc[:,3],1) == VDS)].iloc[:,5].reset_index(drop=True)   ### Change accordingly to your data file
            Igs = df[(round(df.iloc[:,3],1) == VDS)].iloc[:,6].reset_index(drop=True)   ### Change accordingly to your data file
        
            dx = np.gradient(Vgs)
            dy = np.gradient(Ids)
            gm = dy/dx

            TransfDict['Vgs'].append(Vgs)
            TransfDict['devi'].append(devi)
            TransfDict['Ids'].append(Ids)           
            TransfDict['Igs'].append(Igs)
            TransfDict['gm'].append(gm)

    ### Lin ###
    plotter = Plotter(S.Vgslbl, S.Idslbl, S.VUn, S.mIUn)
    plotter.set_folder('Graphs/Trend')
    for i, devi in enumerate(TransfDict['devi']):
        plotter.add_plot(TransfDict['Vgs'][i], TransfDict['Ids'][i]/1e-3, label=f'{round(W[i]/1e-6)} µm')
    plotter.plot(legend_title=S.Vdslbl+f' = {VDS} V \n'+S.Widlbl+' =', save_name=f'Trend_TransfLin_Vds{VDS}')

    ### Log ###
    plotter = Plotter(S.Vgslbl, S.Idslbl, S.VUn, S.IUn)
    plotter.set_folder('Graphs/Trend')
    for i, devi in enumerate(TransfDict['devi']):
        plotter.add_plot(TransfDict['Vgs'][i], abs(TransfDict['Ids'][i]), label=f'{round(W[i]/1e-6)} µm')
    plotter.plot(ylog_scale=True, legend_title=S.Vdslbl+f' = {VDS} V \n'+S.Widlbl+' =', save_name=f'Trend_TransfLog_Vds{VDS}')

    ### Transcond ###
    plotter = Plotter(S.Vgslbl, S.gmlbl, S.VUn, S.mCondUn)
    plotter.set_folder('Graphs/Trend')
    for i, devi in enumerate(TransfDict['devi']):
        plotter.add_plot(TransfDict['Vgs'][i][:int(len(Vgs) / 2)], TransfDict['gm'][i][:int(len(Vgs) / 2)]/1e-3, label=f'{round(W[i]/1e-6)} µm')
    plotter.plot(legend_title=S.Vdslbl+f' = {VDS} V \n'+S.Widlbl+' =', save_name=f'Trend_Transc_Vds{VDS}')

################ Threshold voltage ################
def Threshold(fileNames, VDS):
    ''' Fits the sqrt(Ids) x Vgs, finding automatically the best linear region and return a output file with the threshold voltages '''
    
    ### Output file ###
    fileOut = 'Threshold/Threshold_All.csv'
    f  = open(fileOut,'w')
    f.write('devi,batc,Meas,Vd,Vt\n')

    for file in fileNames:

        df = pd.read_csv(file, skiprows=3, sep='\t')                                    ### Change accordingly to your data file
        devi = fileNames[file][0] # device
        batc = fileNames[file][1] # batch
        Meas = fileNames[file][2] # measurement number

        threshold_voltages = []

        for Vds in VDS:
            
            Ids = df[(round(df.iloc[:,3],1) == Vds)].iloc[:,4].reset_index(drop=True)   ### Change accordingly to your data file
            Vgs = df[(round(df.iloc[:,3],1) == Vds)].iloc[:,5].reset_index(drop=True)   ### Change accordingly to your data file
            
            Vgs = Vgs[:int(len(Vgs)/2)]
            Ids = Ids[:int(len(Ids)/2)]
            ids_sqrt = np.sqrt(np.abs(Ids))

            # Find the most linear region
            model, vgs_linear, ids_sqrt_linear = find_linear_region(Vgs, ids_sqrt)

            # Extract threshold voltage (Vth) where sqrt(abs(Ids)) = 0
            vth = -model.intercept_ / model.coef_[0]
            threshold_voltages.append((Vds, vth))

            plotter = Plotter(S.Vgslbl, S.sqIdslbl, S.VUn, S.sqrtIUn)
            plotter.set_folder('Graphs/TransfSqrt')
        
            textstr = (r'$V_{T} = $ %.3f $V$' % (vth))
        
            plotter.add_plot(Vgs, ids_sqrt, plot_type='scatter', label=S.explbl)
            plotter.add_plot(vgs_linear, model.predict(vgs_linear.reshape(-1, 1)), color='red', annotate_text=textstr, annotate_pos=(0.1,0.5), annotate_cords='axes fraction', label=S.fitlbl)
            plotter.plot(legend_title=S.Vdslbl+f' = {Vds} V', save_name=f'Transfsqrt_{devi}_{batc}_{Meas}_Vds{Vds}V')

            f.write('{},{},{},{},{}\n'.format(devi,batc,Meas,Vds,vth))

    f.close()

################ Threshold voltage trend ################
def TrendThreshold(VDS, W, fileNames):
    ''' Plots the threshold voltage in respect to the channel width '''
    
    for file in fileNames:
        
        df = pd.read_csv(file)

        Vt = df[(df.iloc[:,2] == 'M3') & (df.iloc[:,3] == VDS)].iloc[:,4]

        plotter = Plotter(S.Widlbl, S.Vtlbl, S.ulenUn, S.VUn)
        plotter.set_folder('Graphs/Trend')

        plotter.add_plot(W/1e-6, Vt, plot_type='scatter', label=S.Vdslbl+f' = {VDS} V')
        plotter.plot(save_name=f'Trend_Vt_Vds{VDS}', close=True)

    
################ Transient #################
## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ##
##   Still need to improve this section   ##
## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ##

def Transient(fileNames, IgRange, IdRange, FIT, Mat):
    ''' Plots transient curves with a choice of fiting '''
    ''' Still need to be improved! '''
    
    plotter = Plotter(S.timelbl, S.Idslbl, S.timeUn, S.mIUn)
    plotter.set_folder('Graphs/Transient')

    aux = []

    for file in fileNames:

        df = pd.read_csv(file, skiprows=15)

        Vds, Vgs, dt = fileNames[file]

        idx0  = np.where(df.iloc[:,0] == 0)[0][0]
        idxf  = np.where(df.iloc[:,0] == dt)[0][0]
        
        time = df.iloc[idx0:idxf,0]
        Igs  = df.iloc[idx0:idxf,2]      * IgRange
        Ids  = df.iloc[idx0:idxf,3]      * IdRange
        Ids0 = np.mean(df.iloc[idx0-5001:idx0-1] * IdRange)

        TIME = df.iloc[:,0]
        IDS  = df.iloc[:,3] * IdRange
        
        Qion = simpson(Igs, x=time)

        plotter.add_plot(time, Ids/1e-3, label=S.Vgslbl+f'= {Vgs} V')
        plotter.plot(legend_title=S.Vdslbl+f'= {Vds} V', xlog_scale=True, save_name=f'Idsxt_Log_Vds{Vds}_Vgs{Vgs}', close=True)
        
        plotter.add_plot(TIME, IDS/1e-3, plot_type='line_scatter', label=S.Vgslbl+f'= {Vgs} V')
        plotter.plot(legend_title=S.Vdslbl+f'= {Vds} V', save_name=f'Idsxt_Log_Vds{Vds}_Vgs{Vgs}_FULL', close=True)
        if FIT == None:
            aux.append([Vgs,Vds,Qion])

        elif FIT == 'ExpDec':
            Tau, err = Exp.ExpDec(time, Igs, Vds, Vgs)
            aux.append([Vgs,Vds,Qion,Tau])

        elif FIT == 'FariaDuong':
            Rd,Rs,Cd,gm,f = Exp.FariaDuong(time, Ids, Igs, Ids0, Vds, Vgs)
            aux.append([Vgs,Vds,Qion,Rd,Rs,Cd,gm,Ids0,f])
    
    dfAux = pd.DataFrame(aux)

    plotter = Plotter(S.Vgslbl, S.Qionlbl, S.VUn, S.ChargUn)
    plotter.set_folder('Graphs/Transient/Par')

    for vds in dfAux.iloc[:,1].unique():
        plotter.add_plot(dfAux[dfAux.iloc[:,1] == vds].iloc[:,0], abs(dfAux[dfAux.iloc[:,1] == vds].iloc[:,2]), plot_type='line_scatter', label=f'{vds} V')
    plotter.plot(legend_title=S.Vdslbl+' =', save_name=f'{Mat}_QionxVgs', close=True)

    if FIT == 'ExpDec':

        plotter = Plotter(S.Vgslbl, S.taulbl, S.VUn, S.mtimeUn)
        plotter.set_folder('Graphs/Transient/Par')

        for vds in dfAux.iloc[:,1].unique():
            plotter.add_plot(dfAux[dfAux.iloc[:,1] == vds].iloc[:,0], dfAux[dfAux.iloc[:,1] == vds].iloc[:,3]/1e-3, plot_type='line_scatter',label=f'{vds} V')
        plotter.plot(legend_title=S.Vdslbl+' =', save_name=f'{Mat}_TauxVgs', close=True)

    elif FIT == 'FariaDuong':

        plotter = Plotter(S.Vgslbl, S.Rdlbl, S.VUn, S.MResUn)
        plotter.set_folder('Graphs/Transient/FariaDuong/Par')

        for vds in dfAux.iloc[:,1].unique():
            plotter.add_plot(dfAux[dfAux.iloc[:,1] == vds].iloc[:,0], dfAux[dfAux.iloc[:,1] == vds].iloc[:,3]/1e9, plot_type='line_scatter', label=f'{vds} V')
        plotter.plot(legend_title=S.Vdslbl+' =', save_name=f'{Mat}_RdxVgs', close=True)

        plotter = Plotter(S.Vgslbl, S.Rslbl, S.VUn, S.kResUn)
        plotter.set_folder('Graphs/Transient/FariaDuong/Par')

        for vds in dfAux.iloc[:,1].unique():
            plotter.add_plot(dfAux[dfAux.iloc[:,1] == vds].iloc[:,0], dfAux[dfAux.iloc[:,1] == vds].iloc[:,4]/1e3, plot_type='line_scatter', label=f'{vds} V')
        plotter.plot(legend_title=S.Vdslbl+' =', save_name=f'{Mat}_RsxVgs', close=True)

        plotter = Plotter(S.Vgslbl, S.Cdlbl, S.VUn, S.uCapUn)
        plotter.set_folder('Graphs/Transient/FariaDuong/Par')

        for vds in dfAux.iloc[:,1].unique():
            plotter.add_plot(dfAux[dfAux.iloc[:,1] == vds].iloc[:,0], dfAux[dfAux.iloc[:,1] == vds].iloc[:,5]/1e-6, plot_type='line_scatter', label=f'{vds} V')
        plotter.plot(legend_title=S.Vdslbl+' =', save_name=f'{Mat}_CdxVgs', close=True)

        plotter = Plotter(S.Vgslbl, S.gmlbl, S.VUn, S.mCondUn)
        plotter.set_folder('Graphs/Transient/FariaDuong/Par')

        for vds in dfAux.iloc[:,1].unique():
            plotter.add_plot(dfAux[dfAux.iloc[:,1] == vds].iloc[:,0], dfAux[dfAux.iloc[:,1] == vds].iloc[:,6]/1e-3, plot_type='line_scatter', label=f'{vds} V')
        plotter.plot(legend_title=S.Vdslbl+' =', save_name=f'{Mat}_gmxVgs', close=True)

        plotter = Plotter(S.Vgslbl, S.ffaclbl, S.VUn, S.ArbUn)
        plotter.set_folder('Graphs/Transient/FariaDuong/Par')

        for vds in dfAux.iloc[:,1].unique():
            plotter.add_plot(dfAux[dfAux.iloc[:,1] == vds].iloc[:,0], dfAux[dfAux.iloc[:,1] == vds].iloc[:,8], plot_type='line_scatter', label=f'{vds} V')
        plotter.plot(legend_title=S.Vdslbl+' =', save_name=f'{Mat}_fxVgs', close=True)
