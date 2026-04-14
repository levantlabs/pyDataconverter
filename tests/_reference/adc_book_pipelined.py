"""
Vendored reference for validation of pyDataconverter's PipelinedADC.

SOURCE:   https://github.com/manar-c/adc_book (python/pipelinedADC.py)
IMPORTED: 2026-04-13
AUTHOR:   Manar El-Chammas (attribution preserved)

This file is READ-ONLY. It is used exclusively by
tests/test_pipelined_adc_vs_reference.py to cross-check the pyDataconverter
implementation against the vetted original. Do not modify it — any bug found
here should be investigated in the upstream repository. If the upstream
changes, re-vendor the relevant slice and update the IMPORTED date above.

Classes included: PipelinedADC, Stage, subADC, subDAC, sumGain.
Nothing else from the upstream file is needed.
"""

import numpy as np


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
        self.FSR = FSR
        self.N = N
        self.LSB = self.FSR / N
        self.ref = np.arange(N)/(N-1)*(FSR-self.LSB) - (FSR/2-self.LSB/2)
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
        self.dacout = np.arange(N+1)/(N-1)*(FSR-self.LSB*1) - (FSR/2 + 0*self.LSB)
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
