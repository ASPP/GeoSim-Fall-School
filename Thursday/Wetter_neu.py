import numpy as np

#d_header = np.genfromtxt('produkt_klima_Tageswerte_20071101_20131231_13675.txt',delimiter=";",names=True)
#infile =open('produkt_klima_Tageswerte_20071101_20131231_13675.txt','r')
#firstLine = infile.readline()
a=np.genfromtxt('produkt_klima_Tageswerte_20071101_20131231_13675.txt',delimiter=";",skip_footer=1,skip_header = 1)

a_shape =  a.shape[0]
a_header=np.genfromtxt('produkt_klima_Tageswerte_20071101_20131231_13675.txt',delimiter=";",skip_footer=1+a_shape,dtype=str)

#zip reads rowwise, therefore we need to transpose a
a_dict = dict(zip(list(a_header),a.T))


#d=np.loadtxt('produkt_klima_Tageswerte_20071101_20131231_13675.txt',delimiter=";",skiprows=1)
