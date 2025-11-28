import pandas as pd
import numpy as np


data = pd.DataFrame({"Random Numbers": np.random.randint(1, 101, 10)})


data.to_csv("numbers.csv", index=False)


print(pd.read_csv("numbers.csv"))
