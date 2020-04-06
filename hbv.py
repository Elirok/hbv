import pandas as pd
import numpy as np

class model():
    input_head = ['timestemp', 'precip', 'maxtemp', 'mintemp', 'epot', 'qobs']
    def __init__(self, df_input, parafile):

        self.days = df_input.shape[0]
        self.input = df_input[self.input_head[:5]]
        self.input['meantemp'] = self.input[['maxtemp','mintemp']].mean(axis=1)
        self.qobs = df_input[self.input_head[-1]]

        self.para = dict()
        with open(parafile, 'rb') as f:
            for line in f.readlines()[7:143]:
                values = line.split()
                self.para[values[0].decode("utf-8")] = float(values[4])

        self.zone = {id: snow(id, self.para, self.days) for id in range(10)}
        self.prc = np.zeros([self.days, 10])
        self.tmp = np.zeros([self.days, 10])

        self.fldprecip = np.zeros(self.days)
        self.fldtemper = np.zeros(self.days)
        self.fldsnwpck = np.zeros(self.days)
        self.fldsnwwet = np.zeros(self.days)
        self.fldsnwout = np.zeros(self.days)
        self.fldsnwcov = np.zeros(self.days)

        self.duz = np.zeros(self.days)
        self.soilmst = np.zeros(self.days)
        self.srfrunoff = np.zeros(self.days)
        self.qsim = np.zeros(self.days)

        self.srfwatstor = np.zeros(self.days)
        self.grndwatstor = np.zeros(self.days)
        self.simulate()


    def set_met(self, day, id):
        precip = self.input['precip'].loc[day]
        temp = self.input['meantemp'].loc[day]
        if self.input['precip'].loc[day] > 0:  tmplapse = self.para['TPGRAD']
        else:   tmplapse = self.para['TCGRAD']
        self.tmp[day,id] = temp + tmplapse * (self.zone[id].elev - self.para['ELEVTMP']) / 100

        if temp >= self.para['TX']: pcorr = self.para['RCORR']
        else: pcorr = self.para['RCORR'] * self.para['SCORR']
        self.prc[day,id] = precip * pcorr * (1 + self.para['PGRAD'] * (self.zone[id].elev - self.para['ELEVPRC']) / 100)

    def simulate(self):

        for day in range(self.days):
            for id in range(10):
                self.set_met(day, id)
                #TODO: max/min temp
                self.zone[id].calc_dst(day, self.tmp[day, id], self.prc[day, id])
                self.fldsnwpck[day] += self.zone[id].snwpck[day]
                self.fldsnwwet[day] += self.zone[id].snwwet[day]
                self.fldsnwout[day] += self.zone[id].snwout[day]
                self.fldsnwcov[day] += self.zone[id].snwcov[day]

            self.fldsnwpck[day] = self.fldsnwpck[day]/10
            self.fldsnwwet[day] = self.fldsnwwet[day]/10
            self.fldsnwout[day] = self.fldsnwout[day]/10
            self.fldsnwcov[day] = self.fldsnwcov[day]/10
            self.fldprecip[day] = self.prc[day, :].mean()
            self.fldtemper[day] = self.tmp[day, :].mean()

            self.soil_moisture(day)
            self.response_routines(day)

        self._nse()
        self._frame()

    def soil_moisture(self, day):
        if day == 0: soilmst = self.para['SOLMST']
        else: soilmst = self.soilmst[day-1]

        #TODO: define epot
        ea = (1-self.fldsnwcov[day]) * self.input['epot'].loc[day] * np.min([1, soilmst / self.para['FCDEL']])  # * cov
        self.duz[day] = self.fldsnwout[day] * (soilmst/self.para['FC']) ** self.para['BETA']
        self.soilmst[day] = np.max([soilmst + self.fldsnwout[day] - self.duz[day] - ea, 0.1])

        if self.soilmst[day] > self.para['FC']:
            self.srfrunoff[day] = self.soilmst[day] - self.para['FC']
            self.soilmst[day] = self.para['FC']

    def response_routines(self, day):
        if day == 0:
            perc = np.min([self.para['SRFWATSTOR'] + self.duz[day], self.para['PERC']])
            uz = self.para['SRFWATSTOR'] + self.duz[day] - perc
            lz = self.para['GRNDWATSTOR'] + perc +\
                 (self.fldprecip[day] - self.input['epot'].loc[day]) * self.para['SJODEL'] / 100
        else:
            perc = np.min([self.para['SRFWATSTOR'] + self.duz[day], self.para['PERC']])
            uz = self.srfwatstor[day-1] + self.duz[day] - perc
            lz = self.grndwatstor[day-1] + perc +\
                 (self.fldprecip[day] - self.input['epot'].loc[day]) * self.para['SJODEL'] / 100

        # Upper zone
        q10 = self.para['KUZ'] * uz
        q11 = self.para['KUZ1'] * np.max([0, uz - self.para['UZ1']])
        q12 = self.para['KUZ2'] * np.max([0, uz - self.para['UZ2']])
        q1 = (q10 + q11+ q12) /24
        q2 = self.para['KUZ'] * lz / 24

        self.qsim[day] = (q1 + q2 + self.srfrunoff[day]) * np.power(10, 3) * self.para['FLDAREA'] /(24*3600)
        self.srfwatstor[day] = uz - q1
        self.grndwatstor[day] = lz - q2

    def _nse(self):
        self.nse = 1 - np.sum(np.power((self.qsim - self.qobs), 2))\
                   / np.sum(np.power(self.qobs - self.qobs.mean(), 2))

    def _frame(self):
        output = {
            'TEMP': self.fldtemper,
            'PRECIP': self.fldprecip,
            'SNWPCK': self.fldsnwpck,
            'SNWET': self.fldsnwwet,
            'SNWCOV': self.fldsnwcov,
            'SNWOUT': self.fldsnwout,
            'QOBS': self.qobs,
            'QSIM': self.qsim
            }

        self.output = pd.DataFrame.from_dict(output)
        self.output = self.output.set_index(pd.to_datetime(self.input['timestemp']).dt.date)

class snow:
    quartile = ['00', '25', '50', '75', '100']
    def __init__(self, id, para, days):
        self.id = id
        self.elev = np.average([para[f'ELEV{self.id}'], para[f'ELEV{self.id+1}']])
        self.area = para[f'AREA{self.id+1}']
        self.glac = para[f'GLAC{self.id+1}']
        self.cfr = para['CFR']
        self.lw = para['LW']
        self.ndag = para['NDAG']
        self.cbre = para['CBRE']
        self.tx = para['TX']

        if self.id+1 <= para['NEDNIV']:
            self.forrest = True
            self.maxdst = np.array([para[f'SL{n}'] for n in self.quartile])
            self.cx = para['CXN']
            self.ts = para['TSN']
        else:
            self.forrest = True
            self.maxdst = np.array([para[f'S{n}'] for n in self.quartile])
            self.cx = para['CX']
            self.ts = para['TS']

        # kan justere oppløsningen av dst
        self.maxdst = np.flip(self.maxdst)
        self.drydst = np.zeros([days, 5])
        self.wetdst = np.zeros([days, 5])
        self.outdst = np.zeros([days, 5])

        self.snwpck = np.zeros(days)
        self.snwwet = np.zeros(days)
        self.snwout = np.zeros(days)
        self.snwcov = np.zeros(days)

        self.snwdry0 = para[f'SNWDRY{self.id+1}'] * self.maxdst
        self.snwwet0 = para[f'SNWWET{self.id+1}']
        #self.icebal0 = para[f'ICEBAL{self.id+1}']

    def calc_dst(self, day, temp, precip):
        #TODO: iceing
        if day == 0:
            self.wetdst[day, :] = self.snwwet0
            self.drydst[day, :] = self.snwwet0
        else:
            self.wetdst[day, :] = self.wetdst[day-1, :]
            self.drydst[day, :] = self.drydst[day-1, :]

        if temp < self.tx: snow, rain = precip, 0
        else: rain, snow = precip, 0
        self.drydst[day, :] += snow * self.maxdst
        self.wetdst[day, :] += rain

        for n in range(5):
            if temp < self.ts: frz, smlt = np.min([self.cfr * self.cx * (self.ts - temp), self.wetdst[day,n]]), 0
            else: smlt, frz = np.min([self.cx * (temp - self.ts), self.drydst[day,n]]), 0

            self.wetdst[day,n] += smlt - frz
            self.drydst[day,n] += frz - smlt
            self.outdst[day,n] = np.max([self.wetdst[day,n] - self.drydst[day,n] * self.lw, 0])
            self.wetdst[day,n] = self.wetdst[day, n] - self.outdst[day, n]

        self.snwwet[day] = self.wetdst[day,:].mean()
        self.snwpck[day] = self.drydst[day,:].mean() + self.snwwet[day]
        self.snwout[day] = self.outdst[day,:].mean()
        self.snwcov[day] = self._snwdist(day)

    def _snwdist(self,day):
        v = np.trim_zeros(self.drydst[day, :])
        if len(v) == 0:     return 0
        elif len(v) == 5:   return 1
        elif len(v) == 1:
            if day == 0:    return 0.25
            else:           return self.snwcov[day-1] * 0.25
        else:               return abs(1 / (np.average(np.diff(v / np.max(v))) * 4))
