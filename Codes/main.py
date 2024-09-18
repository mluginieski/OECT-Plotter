########################################################################################
#                                                                                      #
#  Code for plotting experimental curves of OECTs and Inverters: IxV, Output           #
#  transconductance, On/Off, Vt, VTCs, Gain and transient response                     #
#                                                                                      #
#  Data source: SweepMe and Keithley 2612b                                             #
#                                                                                      #
#  Author: Msc Marcos Luginieski                                                       #
#  Contact: mluginieski@gmail.com                                                      #
#  Date: Sep. 18, 2024                                                                 #
#                                                                                      #
########################################################################################

import numpy as np
import PlotterMain as plo

#####################################
#
### File name dictionary structure (valid for IxV, Output and Transf)
#
# {'folder/your_file_name.txt' : ['device', 'Sample', 'Extrainfo']}
# see that everything here is a string
#
################ IxV ################

fileNames = {'IxV_fileName_001.csv'    : ['#1_T1', 'ML', 'M1'],
             'IxV_fileName_002.csv'    : ['#1_T1', 'ML', 'M2']}

plt.IxV(fileNames)                                             # Plots lin and log scale IxV curves and display the resistance


################ Output ################

fileNames = {'Output_fileName_001.csv'    : ['#1_T1', 'ML', 'M1'],
             'Output_fileName_002.csv'    : ['#1_T1', 'ML', 'M2']}

VGS       = [0.0, 0.2, 0.4, 0.6, 0.8]

plo.Output(VGS, fileNames)                                      # Plots a set of drain current for the output curves


################ Transf ################

fileNames = {'Transf_fileName_001.csv'    : ['#1_T1', 'ML', 'M1'],
             'Transf_fileName_002.csv'    : ['#1_T1', 'ML', 'M2']}

VDS       = [0.0, 0.2, 0.4, 0.6]                               # For diffenrent drain voltages. If just one, use VDS = [value]

W = np.array([150e-6, 200e-6, 250e-6, 300e-6, 350e-6, 400e-6]) # For compare devices with different channel width

plo.Transf(VDS, fileNames)                                     # Plot transfer curves in linear and log scale, plot the set of all curves of a single device
                                                               # transconductance, max tranconductance and OnOff ratio (of a single device)
                                                               # Returnn a data file with all gm,max and OnOff off all data from fileNames

plo.Threshold(fileNames, VDS)                                  # Perform the linear fitting of the sqrt(Ids) x Vgs curve and find the Threshold voltage
                                                               # Takes use of the LinearModel.py, finding automatically the best linear region
                                                               # Returns an output file with all Vt of different devices from fileNames dictionary.

plo.TrendThreshold(0.6, W, ['Threshold/Threshold_All.csv'])    # Plot the Threshold voltage x channel width for a given drain voltage.
                                                               # Usage: plo.TrendThreshold(Vds, W, fileName) where W is a np.array and fileName is list of strings

plo.TrendPar(0.6, W, ['Outfile_Transf-1.csv'])                # Plot some OECT parameters against the channel width for a given drain voltage. Graphs included: transcond,max x W, OnOff x W
                                                              # Usage: plo.TrendPar(Vds, W, fileName) where W is a np.array and fileName is list of strings

plo.TrendTransf(0.6, W, fileNames)                            # Plot the transfer curves (lin and log) and transconductance x channel width for a given drain voltage.
                                                              # Usage: plo.TrendTransf(Vds, W, fileName) where W is an np.array and fileName is list of strings


################ VTC ################
#
### File name dictionary structure (valid for VTCs)
#
# {'folder/your_file_name.txt' : ['device_sample', Vdd, load resistance]}
# see that Vdd and load resistance are float numbers.
#

fileNames = {'VTC_fileName_001.csv'   : ['T4_ML', 0.2, 1e3],
             'VTC_fileName_002.csv'   : ['T4_ML', 0.3, 1e3],
             'VTC_fileName_003.csv'   : ['T4_ML', 0.4, 1e3],
             'VTC_fileName_004.csv'   : ['T4_ML', 0.2, 5e3],
             'VTC_fileName_005.csv'   : ['T4_ML', 0.3, 5e3],
             'VTC_fileName_006.csv'   : ['T4_ML', 0.4, 5e3]}
        
plo.VTC(fileNames)                                           # Plot the VTC curves and Gain curves for a give load resistance
                                                             # Usage: plo.VTC(fileNames)
