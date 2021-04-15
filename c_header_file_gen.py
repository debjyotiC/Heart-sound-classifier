import pandas as pd
from decimal import Decimal

df = pd.read_csv("data/fft-data-test.csv").dropna().reset_index(drop=True)

x_test = df.drop(columns=['Frequency']).to_numpy(dtype='float64')
data_selector = 0

# print(x_test[data_selector][75])


for i in range(len(x_test[data_selector])):
    try:
        value, power = str(x_test[data_selector][i]).split('e-')
        str_data = "{val:.2f}*pow(10,{pow})".format(val=int(value), pow=power)
        print(str_data, end=', \n')
    except ValueError:
        str_data_1 = str(x_test[data_selector][i])
        print(str_data_1, end=', \n')






# write_str = "#ifndef FFT_H \n" \
#             "#define FFT_H \n" + \
#             "double data[]={"+"".join("{value}*pow(10, {power}),".format(value, power)
#                                       for value, power in str(x_test[data_selector]).split('e')) + "}; \n" + \
#             "#endif FFT_H"
# c_file = open("data/fft-data.h", 'w')
# c_file.write(write_str)
# c_file.close()

