import matplotlib.pyplot as plt
import pandas as pd

# create data
df = pd.DataFrame([['Raspberry Pi Pico', 1213, 45.30], ['Arduino Nano 33 BLE Sense', 618, 23.19],
                   ['Teensy 4.0', 123, 1.6]], columns=['Board', 'Feature extraction', 'Inference latency'])

df.plot(x='Board', kind='bar', stacked=True)
plt.ylabel("Total inference time (ms)")
plt.xticks(rotation=0)
plt.show()
