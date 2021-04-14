import pandas as pd

df = pd.read_csv("data/fft-data-test.csv").dropna().reset_index(drop=True)

x_test = df.drop(columns=['Frequency']).to_numpy(dtype='float64')
data_selector = 0

print(x_test[data_selector])


write_str = "#ifndef FFT_H \n#define FFT_H \n" + \
            "double data[]={"+"".join("{},".format(i) for i in x_test[data_selector]) + "}; \n" + \
            "#endif FFT_H"
c_file = open("data/fft-data.h", 'w')
c_file.write(write_str)
c_file.close()