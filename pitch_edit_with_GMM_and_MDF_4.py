# -*- coding= utf-8 -*-

"""
Simple pitch estimation
"""

from __future__ import print_function, division
import os
import numpy as np
import scipy.fftpack
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import correlate
from scipy import signal
__author__ = "Jose A. R. Fonollosa"

#Author: Ponç Palau
def gaussian_function(mu, variance, val):
    exponent_num = abs((val-mu)**2)
    exponent_den = 2*variance
    return np.exp(-1*exponent_num/exponent_den)

#Author: Ponç Palau
def magnitude_difference_function(frame, sfreq, cont, vo_unvo):
    """Estimate pitch using magnitudedifference function
    """
    if(vo_unvo[cont]==1):
        frame = frame.astype(np.float)
        frame -= frame.mean()
        amax = np.abs(frame).max()
        if amax > 0:
            frame /= amax
        else:
            return 0
        frame_length = len(frame)

        res = 1000*np.ones((frame_length,1), dtype = np.int32)
        for i in range(4, frame_length-50):
            res[i] = np.sum(np.abs(frame - np.roll(frame,i)))

        peak = np.argmin(res)
        f0 = sfreq / peak

        if f0 > 50 and f0 < 400:
            return f0
        else:
            return 0;
    else:
        return 0;


def wav2f0(options, gui):
    with open(gui) as f:
        contador_arxiu = 0
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            filename = os.path.join(options.datadir, line + ".wav")
            f0_filename = os.path.join(options.datadir, line + ".f0")
            print("Processing:", filename, '->', f0_filename)
            sfreq, data = wavfile.read(filename)
            contador_frame=0
            with open(f0_filename, 'wt') as f0file:
                nsamples = len(data)
                # plt.plot(data)
                # plt.show()


                # From miliseconds to samples
                ns_windowlength = int(round((options.windowlength * sfreq) / 1000))
                ns_frameshift = int(round((options.frameshift * sfreq) / 1000))
                ns_padding = int(round((options.padding * sfreq) / 1000))
                ###################################################################################################################
                #Here we do the GMM processing
                frame_length = ns_windowlength
                energy = np.zeros((int(np.ceil((nsamples+2*ns_padding-ns_windowlength+1)/ns_frameshift)), 1),dtype = np.float)
                # for i in range((len(data)//frame_length) -1):
                #     auxi = np.array(data[i*frame_length:i*frame_length+(frame_length-1)])
                #     #Here we have the energy of each frame
                #     energy[i] = sum(abs(auxi**2))
                counter = 0
                for ini in range(-ns_padding, nsamples - ns_windowlength + ns_padding + 1, ns_frameshift):

                    first_sample = max(0, ini)
                    last_sample = min(nsamples, ini + ns_windowlength)
                    frame = data[first_sample:last_sample]
                    energy[counter] = sum(abs(frame**2))
                    counter=counter+1


                #We normalize the energy, otherwise it would have different values for each frame
                energy = energy/(energy.max())


                #We will have 2 classes: - CLASS 1: UNVOICED - CLASS 2: VOICED
                pw1 = 0.3 #Prior belief in unvoiced
                pw2 = 0.7 # Prior belief in voiced
                #This mu1/2 and var1/2 values are set according to the histogram above
                #(diferent realizations were performed in order to give them reasonable values)
                mu1 = 0
                var1 = 0.1
                mu2 = 1
                var2 = 0.3
                #p_w1_x stands for the probability that the frame belongs to class w1 given that we know x
                p_w1_x = np.zeros((len(energy),1), dtype = np.float)
                p_w2_x = np.zeros((len(energy),1), dtype = np.float)

                for j in range(1,2):
                    for i in range(0,len(energy)-1):
                        p_w1_x[i] = (gaussian_function(mu1, var1, energy[i])*pw1)
                        p_w2_x[i] = (gaussian_function(mu2, var2, energy[i])*pw2)
                    mu1 = ((p_w1_x.T).dot(energy))/(np.sum(p_w1_x))
                    mu2 = ((p_w2_x.T).dot(energy))/(np.sum(p_w2_x))
                    pw1 = sum(p_w1_x)/len(p_w1_x)
                    pw2 = sum(p_w2_x)/len(p_w2_x)
                    var1 = ((p_w1_x.T).dot(((energy-mu1)**2)))/np.sum(p_w1_x)
                    var2 = ((p_w2_x.T).dot(((energy-mu2)**2)))/np.sum(p_w2_x)

                #PLOTS TO SEE GMM PERFORMANCE
                # x_axis = np.arange(0, 1.0, 0.001)
                # y1 = pw1*(np.exp(-((x_axis-float(mu1))**2)/(2*float(var1))))
                # y2 = pw2*(np.exp(-((x_axis-float(mu2))**2)/(2*float(var2))))
                # y3 = y1 + y2
                #
                # plt.subplot(211)
                # plt.hist(energy, bins=20)
                # plt.title("P(w1==UNVOICED)=0.3  P(w2==VOICED)=0.7  mu1=0.1  mu2=0.7  var1=0.1  var2=0.3")
                # plt.subplot(212)
                # plt.plot(x_axis, y1)
                # plt.plot(x_axis, y2)
                # plt.title("Gaussian approximation")
                # plt.show()



                # 0 -> UNVOICED / 1 -> VOICED
                vo_unvo = np.zeros((len(energy),1), dtype = np.int32)
                for i in range(0,len(energy)-1):
                    if( (p_w2_x[i]*pw2) > (p_w1_x[i]*pw1) ):
                        vo_unvo[i] = 1 #Mark the frame as VOICED

                contador = 0
                for ini in range(-ns_padding, nsamples - ns_windowlength + ns_padding + 1, ns_frameshift):
                    first_sample = max(0, ini)
                    last_sample = min(nsamples, ini + ns_windowlength)
                    frame = data[first_sample:last_sample]
                    f0 = magnitude_difference_function(frame, sfreq, contador, vo_unvo)
                    contador+=1
                    print(f0, file=f0file)
            contador_frame+=1


def main(options, args):
    wav2f0(options, args[0])

if __name__ == "__main__": #every time the program is executed __name__ = __main__
    import optparse
    optparser = optparse.OptionParser(
        usage='python3 %prog [OPTION]... FILELIST\n' + __doc__)
    optparser.add_option(
        '-w', '--windowlength', type='float', default=32,
        help='windows length (ms)')
    optparser.add_option(
        '-f', '--frameshift', type='float', default=15,
        help='frame shift (ms)')
    optparser.add_option(
        '-p', '--padding', type='float', default=16,
        help='zero padding (ms)')
    optparser.add_option(
        '-d', '--datadir', type='string', default='data',
        help='data folder')

    options, args = optparser.parse_args()

    if len(args) == 0:
        print("No FILELIST provided")
        optparser.print_help()
        exit(-1)

    main(options, args)
