import numpy
import pandas
import matplotlib.pyplot

columnnames= ["normalization", "model", "fidelity", "u", "time", "N", "ups", "downs", "starting", "ending", "t"]
data= pandas.read_csv("maindata.txt", sep= " ", names= columnnames, header= None)
data["t"]= data["t"].str.split()


rows= data[(data["N"]== 6) & (data["fidelity"]>= 0.998) & (data["starting"]== "0,0")]


newdata= pandas.DataFrame(columns= columnnames)
print(newdata)

for index, row in data.iterrows():
    if row["fidelity"]>= 0.999:
        newdata.append(row.values())
print(newdata)