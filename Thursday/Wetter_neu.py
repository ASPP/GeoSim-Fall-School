# -*- coding: utf-8 -*-
import numpy as np
import string
import pylab 
import datetime

def nice_text(a_string):
    a_string_nice = string.strip(a_string)
    a_string_nice = string.capitalize(a_string_nice)
    return a_string_nice


a=np.genfromtxt('produkt_klima_Tageswerte_20071101_20131231_13675.txt',delimiter=";",skip_footer=1,skip_header = 1)

a_shape =  a.shape[0]
a_header=np.genfromtxt('produkt_klima_Tageswerte_20071101_20131231_13675.txt',delimiter=";",skip_footer=1+a_shape,dtype=str)

#Leerzeichen weglesen, Erster buchstabe groß
for i in range(len(a_header)):
    a_header[i] = nice_text(a_header[i])
    
    

#zip reads rowwise, therefore we need to transpose a
a_dict = dict(zip(list(a_header),a.T))

#Datum umwandeln year/month/day
a_dict['Date'] = []

for timestemp in a_dict['Mess_datum']:
    Year = np.int_(timestemp*0.0001)
    Month = np.int_((timestemp - Year*10000)*0.01)
    Day =  np.int_((timestemp-Year*10000-Month*100))
    
    a_dict['Date'].append(datetime.date(Year, Month, Day))
   # =datetime.date(a_dict['Year'][i],a_dict['Month'][i],a_dict['Day'][i])
    #a_dict['Date'][i] = np.datetime64(A)
    
# datetime.date(a_dict['Year'][1],a_dict['Month'][1],a_dict['Day'][1])


#fPlot
pylab.plot(a_dict['Date'],a_dict[a_header[3]])
#ax.axis('tight')
pylab.title(a_header[3])
pylab.xlabel(a_header[1])
pylab.ylabel(a_header[3]);
pylab.show()

#print nice_text('   helloo   ')


