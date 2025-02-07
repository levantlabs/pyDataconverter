#Class for pipelined ADC

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'figure.autolayout': True})

class PipelinedADC:
    def __init__(self, Nstages, B, N, FSR_ADC, FSR_DAC, G, minADCcode, timeResponse=False, SampleRate = 0,tau_comparator=0, tau_amplifier=0):
        self.Nstages = Nstages #Total number of stages
        self.B = B #Total number of bits
        if timeResponse == False:
            self.timeResponse = [timeResponse] * Nstages #Models transients
        else:
            self.timeResponse = timeResponse
        self.FS = SampleRate

        #Start defining stages
        self.stage = []
        for i in range(self.Nstages):
            if self.timeResponse[i] == True:
                tauC = tau_comparator[i]
                tauA = tau_amplifier[i]
            else:
                tauC = 0
                tauA = 0
            self.stage.append(Stage(N[i], FSR_ADC=FSR_ADC[i], FSR_DAC=FSR_DAC[i], G=G[i], ADC_minCode=minADCcode[i], timeResponse = self.timeResponse[i], FS = self.FS, tauC = tauC, tauA=tauA))

        

    def output(self, vin, limit=True, option = 1):
        stage_out = vin #For first stage
        DOUT = 0
        for i in range(self.Nstages):
            #print('***')
            #print(i)
            #print(stage_out)
            temp = self.stage[i].output(stage_out)
            stage_out = temp
            #print('Stage {} DADC = {}, H = {}'.format(i, self.stage[i].DADC, self.stage[i].H))
            if option == 1: #This was the one I used for almost everything else
                DOUT += DOUT*self.stage[i].H + self.stage[i].DADC #-1 is for current structure
            else: #This is the correction equation, use it for trimming.  Evenaully go back and fix code
                DOUT = DOUT * self.stage[i].H + self.stage[i].DADC #Without the summation
            #print('Intermediate DOUT = {}'.format(DOUT))

        #Testing this line to debug onlye
        #This doesn't match the above loop?!?!
        #print('Retrieving: DADC0 = {}, DADC1 = {}'.format(self.stage[0].DADC, self.stage[1].DADC))
        #print('******')
        #DOUT1 = self.stage[0].DADC * self.stage[1].H + self.stage[1].DADC
        #print('DOUT in loop = {}, DOUT1 = {}'.format(DOUT, DOUT1))
        #print('*****')
        if DOUT < 0 and limit:
            DOUT = 0
        elif DOUT > 2**self.B-1 and limit:
            DOUT = 2**self.B - 1
        return DOUT
        
    

                              

        
    

class Stage:
    def __init__(self, N, FSR_ADC=1, FSR_DAC=1, G=2, ADC_minCode = 0, timeResponse=False, FS = 0,tauC=0, tauA=0):
        self.subADC = subADC(N, FSR=FSR_ADC)
        self.subDAC = subDAC(N, FSR=FSR_DAC)
        self.sumGain = sumGain(G=G)
        self.ADC_minCode = ADC_minCode #For mapping
        self.timeResponse = timeResponse
        self.FS = FS
        self.tauC = tauC
        self.tauA = tauA

    

        self.H = G #Can change later

    def output(self, vin):
        self.DADC = self.subADC.output(vin)
        self.DACOUT = self.subDAC.output(self.DADC)
        self.stageoutput = self.sumGain.output(vin, self.DACOUT)

        newmodel = True
        #newmodel = False

        if self.timeResponse: #Modify self.stageoutput based on time needed
            #With metastabilty, one DAC element hasn't been resolved yet
            
            #get comparator that is closed to vin
            delta_vin = np.abs(self.subADC.ref - vin)
            ind_closest_Comp = np.argmin(delta_vin)
            #Get whether input is less than reference or larger
            if self.subADC.ref[ind_closest_Comp] - vin > 0:
                isLarger = True
            else:
                isLarger = False

            #If isLarger == true, then residue is current subDAC LSB lower
            #If isLarger = False, then residue is current subDAC LSB higher

            #vin has almost no impact
            deltaOutput = self.sumGain.G * (vin*0 - self.subDAC.FSR)/self.subDAC.N #is it /2?
            if newmodel:
                self.stageoutput = self.stageoutput - (2*isLarger - 1) * deltaOutput
            #if isLarger:
            #    self.stageOutput = self.stageOutput - deltaOutput
            #else:
            #    self.stageOutput = self.stageOutput + deltaOutput
                
            
            #This comp is the slowest
            #For now, just set the comp output threshold to 0.5 (doesn't really matter)
            Vc = 0.5
            time_regenerate = self.tauC*np.log(Vc / delta_vin[ind_closest_Comp])
            #print(time_regenerate)
            #Remaining time for amplifier
            TR = 1/self.FS/2 - time_regenerate
            #if TR < 1e-10:
            #    print('****')
            #    print(TR)
                #Calculate amplifier gain error from incomplete settling
            Gerror = (1-np.exp(-TR/self.tauA))
            #print(Gerror)
            #Now, finally, this final DAC element kicks in
            #self.stageoutput = self.stageoutput * Gerror
            if newmodel:
                self.stageoutput = self.stageoutput + (2*isLarger - 1) * deltaOutput * Gerror
            else:
                self.stageoutput = self.stageoutput * Gerror
        
        self.DADC = self.DADC+self.ADC_minCode

        return self.stageoutput


class subADC:
    def __init__(self, N, FSR=1):
        #self.B = bits
        self.FSR = FSR

        self.N = N #Number of levels. 2**self.B
        self.LSB = self.FSR / N

        self.ref = np.arange(N)/(N-1)*(FSR-self.LSB) - (FSR/2-self.LSB/2)
        print('ADC reference = {}'.format(self.ref))
        print('ADC difference = {}, with LSB = {}'.format(np.diff(self.ref), self.LSB))

        self.noisesigma = 0

    def output(self,vin):
        if self.noisesigma > 0:
            vin_adc = vin + np.random.randn(1)*self.noisesigma
        else:
            vin_adc  = vin

        return np.sum(vin_adc >self.ref)

class subDAC:
    def __init__(self, N, FSR=1):
        self.N = N
        self.FSR = FSR
        self.LSB = self.FSR / N
        self.dacout = np.arange(N+1)/(N-1)*(FSR-self.LSB*1) - (FSR/2 + 0*self.LSB)#-self.LSB/2)
        print('DAC values = {}'.format(self.dacout))
        print('DAC difference = {}, LSB = {}'.format(np.diff(self.dacout), self.LSB))
        self.error = np.zeros(N+1)

    def output(self, din):
        din_dac = int(din)
        return self.dacout[din_dac] + self.error[din_dac]

    def add_error(self, error):
        self.error = error

class sumGain:
    def __init__(self, G):
        self.G = G

    def output(self, vin, dacout):
        return self.G*(vin-dacout)
        

def ADCanalysis(dout):
    #Analyze metrics

    dout = dout# - np.mean(dout)
    yfft = np.fft.fft(dout)
    NFFT = len(dout)
    yfft_dB = 20*np.log10(np.abs(yfft[0:int(NFFT/2)])+1e-20)
    yfft_dB[0] = -200 #Remove DC

    #plt.figure()
    #plt.plot(yfft_dB)
    indfund = np.argmax(yfft_dB)
    yfft_dBsort = np.sort(yfft_dB)
    SFDR = yfft_dBsort[-1] - yfft_dBsort[-2]
    #Calculate SNDR
    SNDR = yfft_dBsort[-1] -10*np.log10(np.sum(10**(yfft_dBsort[:-1]/10)))
    print(10*np.log10(np.sum(10**(yfft_dBsort[:-1]/10))))
    print('SFDR = {}'.format(SFDR))
    print('SNDR = {}'.format(SNDR))
    #x = sdfsfd

    #Now, do best fit analysis
    yfft_ideal = yfft*0
    yfft_ideal[indfund] = yfft[indfund]
    yfft_ideal[0] = yfft[0]
    yfft_ideal[NFFT-indfund] = yfft[NFFT-indfund]
    print('Maximum index = {}'.format(indfund))
    print(yfft[indfund])
    print(yfft[NFFT-indfund])
    print(yfft[NFFT-indfund-1])
    #x = sdfsdf
    y_ifft = np.fft.ifft(yfft_ideal)
    #plt.figure()
    #plt.plot(np.imag(y_ifft))
    #plt.show()
    yideal = np.real(y_ifft)#np.real(y_ifft)
    #print('Power ideal = {}'.format(np.std(yideal)))
    #print('Power real = {}'.format(np.std(dout)))
    #print('Gain factor = {}'.format(np.sqrt(np.var(dout)/np.var(yideal))))
    #plt.figure()
    #plt.plot(yideal)
    #plt.plot(dout)
    #plt.show()
    #plt.figure()

    #plt.close('all')
    #plt.figure()
    #plt.plot(dout, 1.000*yideal-dout, '*')
    #plt.show()

    #print('Actual error from FFT = {}, with correction = {}'.format(np.std(yideal-dout), np.std(1.0006*yideal-dout)))
    
    #x = sdfsfdsf

    return yideal, yideal-dout, SFDR, SNDR, yfft_dB
    
    #plt.plot(yideal, dout-yideal, '*')
    #plt.title('Error')
    
    
        
if __name__ == "__main__":

    if False:
        ADC = subADC(8)
        DAC = subDAC(8,FSR=0.9)
        GAIN = sumGain(4)
        #print(ADC.ref)
        #print(np.diff(ADC.ref))

        vin = np.arange(1000)/1000*1.2-0.6

        output = np.zeros(len(vin))
        dacout = np.zeros(len(vin))
        gainout = np.zeros(len(vin))
        for idx, v in enumerate(vin):
            output[idx] = ADC.output(v)
            dacout[idx] = DAC.output(output[idx])
            gainout[idx] = GAIN.output(v, dacout[idx])

        plt.figure()
        plt.plot(vin, output)

        plt.figure()
        plt.plot(vin, (vin-dacout)/DAC.LSB)

        plt.plot(vin, gainout)
        plt.title('Residue')

    #DAC = subDAC(N)
    PipelineADC = PipelinedADC(Nstages = 2, B = 12, N = [8, 2**10+2], FSR_ADC=[1, 1], FSR_DAC=[1, 1], G = [4, 1024/2], minADCcode=[-1,0])
    #With different reference
    #PipelineADC = PipelinedADC(Nstages = 2, B = 12, N = [8, 2**10+2], FSR_ADC=[8/10, 1], FSR_DAC=[8/10, 1], G = [4, 1024/2], minADCcode=[0,0])
    #PipelineADC = PipelinedADC(Nstages = 2, B = 12, N = [7, 2**10+2], FSR_ADC=[1, 8/7], FSR_DAC=[1, 1], G = [4, 1024/2], minADCcode=[-1,0])

    #x = sdfsfdsd
    #print(PipelineADC.stage[0])
    #print(PipelineADC.stage[1])
    #With 8 levels
    PipelineADC_DACerror = PipelinedADC(Nstages = 2,
                                        B = 12,
                                        N = [8, 2**10+2],
                                        FSR_ADC=[1,1],
                                        FSR_DAC=[0.995, 1],
                                        G = [4.0, 1024/2],
                                        minADCcode=[-1, 0])

    PipelineADC_DACerror2 = PipelinedADC(Nstages = 2,
                                        B = 12,
                                        N = [8, 2**10+2],
                                        FSR_ADC=[1,1],
                                        FSR_DAC=[1, 1],
                                        G = [4.0, 1024/2],
                                        minADCcode=[-1, 0])

    #With 7 levels
    #PipelineADC_DACerror = PipelinedADC(Nstages = 2,
    #                                    B = 12,
    #                                    N = [7, 2**10+2],
    #                                    FSR_ADC=[1,8/7],
    #                                    FSR_DAC=[0.995, 1],
    #                                    G = [4.0, 1024/2],
    #                                    minADCcode=[-1, 0])

    #PipelineADC_DACerror2 = PipelinedADC(Nstages = 2,
    #                                    B = 12,
    #                                    N = [7, 2**10+2],
    #                                    FSR_ADC=[1,8/7],
    #                                    FSR_DAC=[1, 1],
    #                                    G = [4.0, 1024/2],
    #                                    minADCcode=[-1, 0])

    PipelineADC_DACerror2.stage[0].subDAC.add_error(np.asarray([0, -0.2, 0.3, 0.05, -0.15, 0, 0.3, -0.3, 0])*0.001)
        #0.001*np.asarray([-1, 1, -1, 1, -1, 1, -1, 1, 0]))

    #With 8 levels

    PipelineADC_GAINerror = PipelinedADC(Nstages = 2,
                                        B = 12,
                                        N = [8, 2**10+2],
                                        FSR_ADC=[1,1],
                                        FSR_DAC=[1, 1],
                                        G = [3.988, 1024/2],
                                        minADCcode=[-1, 0])

    #dout_array = np.zeros(len(vin))
    #dout_array_dacerror = np.zeros(len(vin))
    #for idx, v in enumerate(vin):
    #    dout_array[idx] = PipelineADC.output(v)
        #dout_array_dacerror[idx] = PipelineADC_DACerror.output(v)

    #plt.figure()
    #plt.plot(vin, dout_array)

    #Create sine wave
    FSR = 1
    NFFT = 2**16
    NFFT = 2**15 #Updated on 4/11/2023
    NFFTsim = 256
    NFIN = NFFT/128-1
    t = np.arange(NFFT)
    tsim = np.arange(NFFTsim)
    xin = FSR/2*0.95*np.sin(2*np.pi * NFIN/NFFT*t)
    xinsim = FSR/2*0.95 * np.sin(2*np.pi*NFIN/NFFTsim*tsim)

    DOUT_sine = np.zeros(NFFT)
    DOUT_sine_sim = np.zeros(NFFTsim)
    BACKEND_sine = np.zeros(NFFT)
    DOUT_ST1_sine = np.zeros(NFFT)
    ADC1_sine = np.zeros(NFFT)
    DOUT_sine_DACerror = np.zeros(NFFT)
    ADC1_sine_DACerror = np.zeros(NFFT)
    DOUT_sine_DACerror2 = np.zeros(NFFT)
    ADC1_sine_DACerror2 = np.zeros(NFFT)
    DOUT_sine_DACerror2_sim = np.zeros(NFFTsim)
    BACKEND_sine_DACerror = np.zeros(NFFT)
    DOUT_sine_GAINerror = np.zeros(NFFT)
    ADC1_sine_GAINerror = np.zeros(NFFT)
    BACKEND_sine_GAINerror = np.zeros(NFFT)
    for idx, x in enumerate(xin):
        DOUT_sine[idx] = PipelineADC.output(x)
        DOUT_ST1_sine[idx] = PipelineADC.stage[0].DADC*PipelineADC.stage[1].H
        ADC1_sine[idx] = PipelineADC.stage[0].DADC
        DOUT_sine_DACerror[idx] = PipelineADC_DACerror.output(x)
        ADC1_sine_DACerror[idx] = PipelineADC_DACerror.stage[0].DADC #First stage output
        DOUT_sine_GAINerror[idx] = PipelineADC_GAINerror.output(x)
        DOUT_sine_DACerror2[idx] = PipelineADC_DACerror2.output(x)
        ADC1_sine_DACerror2[idx] = PipelineADC_DACerror2.stage[0].DADC #First stage output
        BACKEND_sine[idx] = PipelineADC.stage[1].DADC
        BACKEND_sine_DACerror[idx] = PipelineADC_DACerror.stage[1].DADC
        BACKEND_sine_GAINerror[idx] = PipelineADC_GAINerror.stage[1].DADC
        ADC1_sine_GAINerror[idx] = PipelineADC_GAINerror.stage[0].DADC #First stage output

    for idx, x in enumerate(xinsim):
        DOUT_sine_sim[idx] = PipelineADC.output(x)
        DOUT_sine_DACerror2_sim[idx] = PipelineADC_DACerror2.output(x)

    #Create simple pipeline with 2 levels in first stage
    PipelineADC_example1 = PipelinedADC(Nstages = 2, B = 11, N = [2, 2**10+2], FSR_ADC=[3/4, 1], FSR_DAC=[3/4, 1], G = [1, 1024/2*7/8], minADCcode=[0,0])
    DOUT_ex = np.zeros(NFFT)
    BACKEND_ex = np.zeros(NFFT)
    xinlin = np.linspace(-0.5, 0.5, NFFT)
    for idx, x in enumerate(xinlin):
        DOUT_ex[idx] = PipelineADC_example1.output(x)
        BACKEND_ex[idx] = PipelineADC_example1.stage[1].DADC

    ind_s = np.argsort(xinlin)
    plt.figure()
    plt.plot(xinlin[ind_s], 4*(BACKEND_ex[ind_s]-512)/2**10)
    plt.figure()
    plt.plot(DOUT_ex[ind_s])
    #plt.show()
    #x = sdfsfdxs
        
        
    #plt.figure(figsize=(6.4, 3.2))
    #plt.plot(DOUT_sine)
    if True:
        #Plot output and best fit
        #For simulation
        #d_idealsim, error_nlsim, SFDRsim,SNDRsim = ADCanalysis(DOUT_sine_sim)
        plt.figure(figsize=(6.4, 5.8))
        #plt.subplot(2,1,1)
        #plt.plot(DOUT_sine_sim, error_nlsim, 'k')
        #plt.subplot(2, 1, 2)
        #plt.plot(d_idealsim, error_nlsim, color = '0.4', marker='*')
        #ind_s = np.argsort(d_idealsim)
        #plt.plot(d_idealsim[ind_s], 0.5*np.ones(NFFTsim), 'k-.', linewidth=3)
        #plt.plot(d_idealsim[ind_s], -0.5*np.ones(NFFTsim), 'k-.', linewidth=3)
        #plt.grid('on', linestyle=':')
        #plt.xlabel('(a)')
        #plt.ylabel('Error [LSB]')
        #plt.ylim([-0.6, 0.6])
        #plt.plot(d_idealsim, 'k:')
        #plt.show()
        #x = sdfsfd


        #plt.subplot(2, 1, 2)
        #plt.figure(figsize=(6.4, 3.2))
        d_ideal, error_nl, SFDR, SNDR, yfft_dB_ideal = ADCanalysis(DOUT_sine)
        
        #ind_s = np.argsort(DOUT_sine)
        #Color code based on code
        plt.subplot(2, 1, 1)
        adcunique = np.unique(ADC1_sine)
        for iadc, adc in enumerate(adcunique):
            ind_a = ADC1_sine == adc
            di_temp = d_ideal[ind_a]
            error_temp = error_nl[ind_a]
            ind_s = np.argsort(di_temp)
            if iadc % 2 == 0:
                col = '0.4'
            else:
                col = '0.2'
            
    
            plt.plot(di_temp[ind_s], error_temp[ind_s], color=col)
        ind_s = np.argsort(d_ideal)
        #plt.plot(d_ideal[ind_s], error_nl[ind_s], '0.4')
        #plt.plot(DOUT_sine[ind_s], error_nl[ind_s])
        plt.plot(d_ideal[ind_s], 0.5*np.ones(NFFT), 'k-.', linewidth=3)
        plt.plot(d_ideal[ind_s], -0.5*np.ones(NFFT), 'k-.', linewidth=3)
        plt.grid('on', linestyle=':')
        plt.xlabel('(a)')#Ideal Output [LSB]')
        plt.ylabel('Error [LSB]')
        #plt.ylim([-0.6, 0.6])
        plt.tight_layout()

        #plt.figure()
        #error_backend = (d_ideal+np.mean(DOUT_sine) - DOUT_ST1_sine) - BACKEND_sine
        #plt.plot(d_ideal[ind_s], error_backend[ind_s])
        #plt.title('BACKEND ideal')
        #plt.show()
        plt.savefig('../images/pythonplot/intro_adcnl_ideal.pdf')
        #x = sdfsfd


    d_ideal, error_nl, SFDR, SNDR, yfft_dB_DACerror2 = ADCanalysis(DOUT_sine_DACerror2)
    d_ideal = DOUT_sine_DACerror2
    ind_s = np.argsort(d_ideal)
    #d_ide
    #Color code
    adcunique = np.unique(ADC1_sine_DACerror2)
    plt.subplot(2, 1, 2)
    for iadc, adc in enumerate(adcunique):
        ind_a = ADC1_sine_DACerror2 == adc
        di_temp = d_ideal[ind_a]
        error_temp = error_nl[ind_a]
        ind_s = np.argsort(di_temp)
        if iadc % 2 == 0:
            col = '0.4'
        else:
            col = '0.2'
            
    
        plt.plot(di_temp[ind_s], error_temp[ind_s], color=col)
        #Add text
        #plt.text(min(di_temp), 1.0, 'D1={}'.format(int(adc)))
    ind_s = np.argsort(d_ideal)
    plt.plot(d_ideal[ind_s], 0.5*np.ones(NFFT), 'k-.', linewidth=3)
    plt.plot(d_ideal[ind_s], -0.5*np.ones(NFFT), 'k-.', linewidth=3)
    plt.grid('on', linestyle=':')
    plt.xlabel('Ideal Output [LSB]\n(b)')
    plt.ylabel('Error [LSB]')
    plt.tight_layout()

    plt.savefig('../images/pythonplot/intro_adcnl_ideal.pdf')
    

    #plt.show()
    #x = sdfsdfs
    plt.figure(figsize=(6.4, 5.8))
    plt.subplot(2, 1, 2)

    #plt.plot(DOUT_sine)
    d_ideal, error_nl, SFDR, SNDR, yfft_dB_DACerror = ADCanalysis(DOUT_sine_DACerror)
    adcunique = np.unique(ADC1_sine_DACerror)
    for iadc, adc in enumerate(adcunique):
        ind_a = ADC1_sine_DACerror == adc
        di_temp = d_ideal[ind_a]
        error_temp = error_nl[ind_a]
        ind_s = np.argsort(di_temp)
        if iadc % 2 == 0:
            col = '0.4'
        else:
            col = '0.2'
            
    
        plt.plot(di_temp[ind_s], error_temp[ind_s], color=col)
    ind_s = np.argsort(d_ideal)
    #plt.plot(d_ideal[ind_s], error_nl[ind_s], '0.4')
    plt.plot(d_ideal[ind_s], 0.5*np.ones(NFFT), 'k-.', linewidth=3)
    plt.plot(d_ideal[ind_s], -0.5*np.ones(NFFT), 'k-.', linewidth=3)
    plt.grid('on', linestyle=':')
    plt.xlabel('Ideal Output [LSB]\n(b)')#Ideal Output [LSB]')
    plt.ylabel('Error [LSB]')

    plt.subplot(2,1,1)
    #plt.plot(DOUT_sine)
    d_ideal, error_nl, SFDR, SNDR, ytemp = ADCanalysis(DOUT_sine_GAINerror)
    adcunique = np.unique(ADC1_sine_GAINerror)
    for iadc, adc in enumerate(adcunique):
        ind_a = ADC1_sine_GAINerror == adc
        di_temp = d_ideal[ind_a]
        error_temp = error_nl[ind_a]
        ind_s = np.argsort(di_temp)
        if iadc % 2 == 0:
            col = '0.4'
        else:
            col = '0.2'
            
    
        plt.plot(di_temp[ind_s], error_temp[ind_s], color=col)
    ind_s = np.argsort(d_ideal)
    #plt.plot(d_ideal[ind_s], error_nl[ind_s], '0.4')
    plt.plot(d_ideal[ind_s], 0.5*np.ones(NFFT), 'k-.', linewidth=3)
    plt.plot(d_ideal[ind_s], -0.5*np.ones(NFFT), 'k-.', linewidth=3)
    plt.grid('on', linestyle=':')
    plt.xlabel('(a)')
    plt.ylabel('Error [LSB]')
    plt.tight_layout()
    plt.savefig('../images/pythonplot/intro_adcnl_dacgain_error.pdf')

    plt.figure(figsize=(6.4, 3.2))
    #plt.subplot(2, 1, 1)
    #d_ideal, error_nl, SFDR, SNDR = ADCanalysis(DOUT_sine_DACerror2_sim)
    #plt.plot(d_ideal, error_nl, color='0.4', marker='*')
    #plt.subplot(2, 1, 2)
    d_ideal, error_nl, SFDR, SNDR, yfft_dB_DACerror2 = ADCanalysis(DOUT_sine_DACerror2)
    d_ideal = DOUT_sine_DACerror2
    ind_s = np.argsort(d_ideal)
    #d_ide
    #Color code
    adcunique = np.unique(ADC1_sine_DACerror2)
    for iadc, adc in enumerate(adcunique):
        ind_a = ADC1_sine_DACerror2 == adc
        di_temp = d_ideal[ind_a]
        error_temp = error_nl[ind_a]
        ind_s = np.argsort(di_temp)
        if iadc % 2 == 0:
            col = '0.4'
        else:
            col = '0.2'
            
    
        plt.plot(di_temp[ind_s], error_temp[ind_s], color=col)
        #Add text
        #plt.text(min(di_temp), 1.0, 'D1={}'.format(int(adc)))
    ind_s = np.argsort(d_ideal)
    plt.plot(d_ideal[ind_s], 0.5*np.ones(NFFT), 'k-.', linewidth=3)
    plt.plot(d_ideal[ind_s], -0.5*np.ones(NFFT), 'k-.', linewidth=3)
    plt.grid('on', linestyle=':')
    plt.xlabel('Ideal Output [LSB]')
    plt.ylabel('Error [LSB]')
    plt.tight_layout()
    plt.savefig('../images/pythonplot/intro_adcnl_dacerror.pdf')

    #Plot vs. adc output
    plt.figure(figsize=(6.4, 3.2))
    ind_s = np.argsort(DOUT_sine_DACerror2)
    plt.plot(DOUT_sine_DACerror2[ind_s], error_nl[ind_s]) 
    plt.plot(DOUT_sine_DACerror2[ind_s], 0.5*np.ones(NFFT), 'k-.', linewidth=3)
    plt.plot(DOUT_sine_DACerror2[ind_s], -0.5*np.ones(NFFT), 'k-.', linewidth=3)
    plt.grid('on', linestyle=':')
    plt.xlabel('ADC Output [LSB]')
    plt.ylabel('Error [LSB]')
    plt.tight_layout()
    #plt.savefig('../images/pythonplot/intro_adcnl_dacerror.jpg', dpi=300)

    #plt.show()
    #x = sdfsfd


    #Plot FFT for three cases


    #Recalculate FFT, but with downsampled data

    d_ideala, error_nla, SFDRa, SNDRa, yfft_dB_DACerror2 = ADCanalysis(DOUT_sine_DACerror2)
    d_ideala, error_nla, SFDR, SNDRa, yfft_dB_DACerror = ADCanalysis(DOUT_sine_DACerror)
    d_ideala, error_nla, SFDRa, SNDRa, yfft_dB_ideal = ADCanalysis(DOUT_sine)
    plt.figure(figsize=(6.4, 6.4))
    plt.subplot(3, 1, 1)
    plt.plot(np.linspace(0, 0.5, num = len(yfft_dB_ideal)), yfft_dB_ideal-max(yfft_dB_ideal) - 1, 'k')
    plt.ylim([-120, 0])
    plt.ylabel('PSD [dBFS]')
    plt.xlim([0, 0.5])
    plt.xlabel('(a)')
    plt.grid('on')

    plt.subplot(3, 1, 2)
    plt.plot(np.linspace(0, 0.5, num = len(yfft_dB_DACerror)), yfft_dB_DACerror-max(yfft_dB_DACerror)-1, 'k')
    plt.ylim([-120, 0])
    plt.ylabel('PSD [dBFS]')
    plt.xlabel('(b)')
    plt.xlim([0, 0.5])
    plt.grid('on')

    plt.subplot(3, 1, 3)
    plt.plot(np.linspace(0, 0.5, num = len(yfft_dB_DACerror2)), yfft_dB_DACerror2-max(yfft_dB_DACerror2)-1, 'k')
    plt.ylim([-120, 0])
    plt.ylabel('PSD [dBFS]')
    plt.xlabel('FIN/FS\n(c)')
    plt.xlim([0, 0.5])
    plt.grid('on')
    plt.tight_layout()
    plt.savefig('../images/pythonplot/example_fft.pdf')
    

    #plt.show()
    plt.close('all')


    #x = sdfsdfs
    #analyze pipelined ADC example for MDAC architecture
    xin = FSR/2*1*np.sin(2*np.pi * NFIN/NFFT*t)
    #DAC = subDAC(N)
    PipelineADC = PipelinedADC(Nstages = 2, B = 12, N = [8, 2**10-1], FSR_ADC=[1, 1], FSR_DAC=[1, 1], G = [4, 1024/2], minADCcode=[-1,0])
    #Below is with fixed transfer function, on 2/10/2023
    PipelineADC = PipelinedADC(Nstages = 2, B = 12, N = [8, 2**10], FSR_ADC=[1, 1], FSR_DAC=[1, 1], G = [4, 1024/2], minADCcode=[-1,0])
    #PipelineADC = PipelinedADC(Nstages = 2, B = 12, N = [8, 2**10-1], FSR_ADC=[8/10, 1], FSR_DAC=[8/10, 1], G = [4, 1024/2*8/10], minADCcode=[0,0])
    for idx, x in enumerate(xin):
        DOUT_sine[idx] = PipelineADC.output(x, limit=True, option=2) #Added option = 2 to use fixed calculation
        BACKEND_sine[idx] = PipelineADC.stage[1].DADC

    plt.close('all')
    plt.figure()
    plt.plot(xin, DOUT_sine)

    d_ideal, error_nl, SFDR, SNDR, ytemp = ADCanalysis(DOUT_sine)
    plt.figure()
    ind_s = np.argsort(d_ideal)
    plt.plot(d_ideal[ind_s], error_nl[ind_s])

    plt.figure(figsize=(6.4, 3.2))
    indsort = np.argsort(xin)
    bound = np.ones(len(xin))*0.5

    plt.plot(xin[indsort], (BACKEND_sine[indsort]-1024/2)/1024, '0.4')
    plt.plot(xin[indsort], bound * 1, 'k-.', linewidth=3)
    plt.plot(xin[indsort], bound * 0.5, 'k:', linewidth=1)
    plt.plot(xin[indsort], bound * -1, 'k-.', linewidth=3)
    plt.plot(xin[indsort], bound * -0.5, 'k:', linewidth=1)
    plt.ylim([-0.7, 0.7])
    plt.xlim([-0.5, 0.5])
    plt.yticks([-0.5, -0.25, 0, 0.25, 0.5])
    plt.xticks([-0.5, -0.25, 0, 0.25, 0.5])
    plt.grid('on', linestyle=':')
    plt.legend(['Residue', 'Bound', 'Margin'], ncol = 3, loc='upper right')
    plt.xlabel('Input [FSR]')
    plt.ylabel('Output Residue [V]')
    plt.tight_layout()
    plt.savefig('../images/pythonplot/arch_mdac_n8_residue.pdf')

    #plt.show()
    #x = sdfsdf


    PipelineADC = PipelinedADC(Nstages = 2, B = 12, N = [6, 2**10-1], FSR_ADC=[0.75, 1], FSR_DAC=[0.75, 1], G = [4, 1024/2], minADCcode=[0,0])

    xin2 = FSR/2*1.1*np.sin(2*np.pi * NFIN/NFFT*t)
    DOUT_sine_large = np.zeros(len(xin2))
    for idx, x in enumerate(xin):
        DOUT_sine[idx] = PipelineADC.output(x)
        BACKEND_sine[idx] = PipelineADC.stage[1].DADC
        DOUT_sine_large[idx] = PipelineADC.output(xin2[idx], limit=True)

    plt.figure(figsize=(6.4, 3.2))
    #plt.subplot(2, 1, 1)
    indsort = np.argsort(xin)
    bound = np.ones(len(xin))*0.5

    plt.plot(xin[indsort], (BACKEND_sine[indsort]-1024/2)/1024,'0.4')
    plt.plot(xin[indsort], bound * 1, 'k-.', linewidth=3)
    plt.plot(xin[indsort], bound * -1, 'k-.', linewidth=3)
    plt.grid('on', linestyle=':')
    plt.ylabel('Output Residue [V]')
    plt.xlabel('Input [V]')
    plt.tight_layout()
    plt.savefig('../images/pythonplot/arch_mdac_n6_residue_example.pdf')
    

    plt.figure(figsize=(6.4, 6.4))
    plt.subplot(2, 1, 1)
    indsort = np.argsort(xin)
    bound = np.ones(len(xin))*0.5

    plt.plot(xin[indsort], (BACKEND_sine[indsort]-1024/2)/1024,'0.4')
    plt.plot(xin[indsort], bound * 1, 'k-.', linewidth=3)
    plt.plot(xin[indsort], bound * -1, 'k-.', linewidth=3)
    plt.grid('on', linestyle=':')
    plt.ylabel('Output Residue [V]')
    plt.xlabel('(a)')

    PipelineADC = PipelinedADC(Nstages = 2, B = 12, N = [6, 2**10-1], FSR_ADC=[1, 1], FSR_DAC=[1, 1], G = [4, 1024/2*1/0.75], minADCcode=[-0.75,0])
    DOUT_sine2 = np.zeros(len(xin))
    DOUT_sine2_large = np.zeros(len(xin2))
    for idx, x in enumerate(xin):
        DOUT_sine2[idx] = PipelineADC.output(x)
        BACKEND_sine[idx] = PipelineADC.stage[1].DADC
        DOUT_sine2_large[idx]= PipelineADC.output(xin2[idx], limit=False)


    #plt.figure()
    plt.subplot(2, 1, 2)
    indsort = np.argsort(xin)
    bound = np.ones(len(xin))*0.5

    plt.plot(xin[indsort], (BACKEND_sine[indsort]-1024/2)/1024,'0.4')
    plt.plot(xin[indsort], bound * 1, 'k-.', linewidth=3)
    plt.plot(xin[indsort], bound * -1, 'k-.', linewidth=3)
    plt.grid('on', linestyle=':')
    plt.xlabel('Input [V]\n(b)')
    plt.ylabel('Output Residue [V]')
    plt.tight_layout()
    plt.savefig('../images/pythonplot/arch_mdac_n6_residue.pdf')
    #plt.figure()
    #xplt.plot(DOUT_sine - DOUT_sine2)

    #plt.show()
    #x = sdfsdfsdf

    plt.figure()
    #plt.subplot(2, 1, 1)
    plt.plot(DOUT_sine2_large, 'k:')
    #plt.xlabel('(a)')
    #plt.ylabel('ADC Output')
    #plt.subplot(2, 1, 2)
    plt.plot(DOUT_sine_large, 'k')
    plt.xlabel('Sample')
    plt.ylabel('ADC Output')
    plt.grid('on', linestyle=':')
    #plt.tight_layout()
    plt.legend(['Extended ADC range', 'Limited ADC range'])
    plt.savefig('../images/pythonplot/arch_dout_n6.pdf')

    #Gain error to analyze ADC performance (for cap mismatch
    plt.close('all')


    #Plot impact of N on gain error first

    #
    
    NFFT = 2**14
    NFFTsim = 256
    NFIN = NFFT/128-1
    t = np.arange(NFFT)
    numG = 30
    numG = 60
    xin = FSR/2*0.99*np.sin(2*np.pi * NFIN/NFFT*t)
    #Gain of 4
    G_ideal = 4
    G_array = 4+0.01*(np.arange(numG)/(numG-1) - 0.5)
    G_err = 0.01 * (np.arange(numG)/(numG - 1) - 0.5)
    G_array = G_ideal * (1 + G_err)
    SFDR_array = np.zeros(numG)
    SNDR_array = np.zeros(numG)
    DOUT_sine = np.zeros(NFFT)
    BACKEND_sine = np.zeros(NFFT)
    for jdx,G in enumerate(G_array):
        PipelineADC = PipelinedADC(Nstages = 2, B = 12, N = [8, 2**10+2], FSR_ADC=[1, 1], FSR_DAC=[1, 1], G = [G, 1024/2], minADCcode=[-1,0])
        #Updated on 2/10/2023
        PipelineADC = PipelinedADC(Nstages = 2, B = 12, N = [8, 2**10], FSR_ADC=[1, 1], FSR_DAC=[1, 1], G = [G, 1024/2], minADCcode=[-1,0])
        for idx, x in enumerate(xin):
            DOUT_sine[idx] = PipelineADC.output(x, limit=True, option = 2) #updated on 2/10 to use new math
            #BACKEND_sine[idx] = PipelineADC.stage[1].DADC

        d_ideal, error_nl, SFDR, SNDR, x = ADCanalysis(DOUT_sine)
        print('SFDR ={} with gain = {}'.format(SFDR, G))
        SFDR_array[jdx] = SFDR
        SNDR_array[jdx] = SNDR

    #Gain of 8
    G_ideal = 8
    #G_array = 4+0.01*(np.arange(numG)/(numG-1) - 0.5)
    #G_err = 0.01 * (np.arange(numG)/(numG - 1) - 0.5)
    G_array = G_ideal * (1 + G_err)
    SFDR_array2 = np.zeros(numG)
    SNDR_array2 = np.zeros(numG)
    sigPower = np.zeros(numG)
    for jdx,G in enumerate(G_array):
        PipelineADC = PipelinedADC(Nstages = 2, B = 12, N = [16, 2**9+2], FSR_ADC=[1, 1], FSR_DAC=[1, 1], G = [G, 1024/2/2], minADCcode=[-1,0])
        PipelineADC = PipelinedADC(Nstages = 2, B = 12, N = [16, 2**9], FSR_ADC=[1, 1], FSR_DAC=[1, 1], G = [G, 256], minADCcode=[-1,0])
        #PipelineADC = PipelinedADC(Nstages = 2, B = 12, N = [4, 2**11], FSR_ADC=[1, 1], FSR_DAC=[1, 1], G = [G, 256*2*2], minADCcode=[-1,0])
        for idx, x in enumerate(xin):
            DOUT_sine[idx] = PipelineADC.output(x, limit=True, option=3)
            BACKEND_sine[idx] = PipelineADC.stage[1].DADC

        #plt.plot(DOUT_sine, BACKEND_sine, '*')
        #plt.show()
        #x = sdfsdf
        d_ideal, error_nl, SFDR, SNDR, x = ADCanalysis(DOUT_sine)
        print('SFDR ={} with gain = {}.  Signal power = {}.'.format(SFDR, G, np.std(DOUT_sine)))
        sigPower[jdx] = np.std(DOUT_sine)
        SFDR_array2[jdx] = SFDR
        SNDR_array2[jdx] = SNDR


    print('SFDR = {}'.format(SFDR_array))
    print('Sig power = {}'.format(sigPower))
    plt.figure(figsize=(6.4, 5.8))
    #plt.plot(G_array, SFDR_array)
    plt.subplot(2, 1, 1)
    #plt.plot(G_array - 4, SNDR_array, '0.4')
    plt.plot(G_err*100, SNDR_array, '0.2')
    plt.plot(G_err*100, SNDR_array2, '0.6')

    #anlaysis estimate
    LSB_backend = 1 / 2**10
    LSB_1 = 1/2**12
    LSB_2 = 1/2**12
    LSB_ST1_1 = 1 / 8
    LSB_ST1_2 = 1/16
    #err = np.sqrt(6)*LSB_backend / (LSB_ST1) / 2/np.sqrt(2)
    err = 1 * 1*np.sqrt(1*LSB_1**2 / LSB_ST1_1**2)
    err2 = np.sqrt(LSB_2**2 / LSB_ST1_2**2)
    #err2 = 16/2**12
    
    #plt.subplot(2, 1, 1)
    SNDR_ideal = 6.02*12+1.76
    plt.plot(G_err*100, np.ones(len(G_array)) * np.max(SNDR_ideal-3), 'k:')
    print(err)
    plt.plot((-err*100)*np.ones(10), np.arange(10)+65 , 'k-.')
    plt.plot((-err2*100)*np.ones(10), np.arange(10)+65 , 'k--')
    plt.plot((+err*100)*np.ones(10), np.arange(10)+65, 'k-.')
    
    plt.plot((+err2*100)*np.ones(10), np.arange(10)+65, 'k--')

    plt.legend(['SNDR with G=4', 'SNDR with G=8', '3dB drop', 'Theoretical error bound with G=4', 'Theoretical error bound with G=8'])

    plt.xlabel('(a)')
    plt.ylabel('ADC SNDR [dB]')
    plt.grid('on', linestyle=':')
    plt.subplot(2, 1, 2)
    plt.plot(G_err*100, SFDR_array ,'0.2')
    plt.plot(G_err*100, SFDR_array2, '0.6')
    plt.plot(G_err*100, np.ones(len(G_array))*90, 'k-.')
    plt.plot(G_err*100, np.ones(len(G_array))*85, 'k:')
    plt.legend(['SFDR with G=4', 'SFDR with G=8', '90dB', '85dB'])
    plt.ylabel('ADC SFDR [dB]')
    plt.xlabel('Gain error [%]\n(b)')
    plt.grid('on', linestyle=':')
    plt.tight_layout()
    plt.savefig('../images/pythonplot/arch_gain_capmismatch.pdf')





    plt.show()
