import numpy as np
import pandas as pd
import csv

full = '../DataFiles/FullData.csv'
grouped = '../DataFiles/grouped.csv'

finalF = "FINAL.csv"
F01 = "F01.csv"

full = pd.read_csv(full)
grouped = pd.read_csv(grouped)

cols = list(full.columns)
colsF = cols
cols.append("eTIV")
cols.append("hippoL")
cols.append( "hippoR")

with open(finalF, mode='w') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(cols)
    f.close()

for inx, row in full.iterrows():
    if 'm' not in row['ID'] and 'SH_DARE' not in row['ID']:
        id = row['ID'].replace('bl','')
        for i, r in grouped.iterrows():
            if int(id) == r['ID']:
                print("....")
                x = list(row)
                x.append(r['eTIV'])
                x.append(r['hippoL'])
                x.append(r['hippoR'])
                with open(finalF, mode='a+') as f:
                    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(x)

f.close()


with open(F01, mode='w') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(cols)
    f.close()

final = pd.read_csv(finalF)
final = final.drop_duplicates(subset =["ID"])
full = full.drop_duplicates(subset =["ID"])

for i, r in full.iterrows():

    if str(r['ID']) in final['ID'].values:

        xf = list(final.iloc[(final[final['ID'] == str(r['ID'])].index[0])])

        with open(F01, mode='a+') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(xf)
            f.close()
    else:
        x = list(r)
        with open(F01, mode='a+') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(x)
            f.close()
