KAGGLE_DATA = "../../data/kaggle_data_processed.csv"
OUR_DATA = "../../data/processed_data.txt"

OUR_PREFIX = "processed_data_halfk/"
KAGGLE_PREFIX = "kaggle_additional/"

import pandas as pd
import numpy as np

# load data

# patient_id, L_CC, L_MLO, R_CC, R_MLO, cancer
our_data = pd.read_csv(OUR_DATA, delimiter=",", header=None)
# change cancer columnt to 0 or 1.
# change 100 to 0 and everything else to 1
our_data[5] = np.where(our_data[5] == 100, 0, 1)
# replace all image paths with the image id. mistične lambde so rešitev
for i in range(1, 5):
    our_data[i] = our_data[i].apply(
        lambda x: OUR_PREFIX + x.replace("zdrave/", "").replace("rakave/", "")
    )


# cancer,patient_id,L_CC,L_MLO,R_MLO,R_CC
kaggle_data = pd.read_csv(KAGGLE_DATA, delimiter=",", header=None)
# new column names: patient_id, L_CC, L_MLO, R_CC, R_MLO, cancer
kaggle_data.columns = [5, 0, 1, 2, 4, 3]
# add image prefix
for i in range(1, 5):
    kaggle_data[i] = kaggle_data[i].apply(lambda x: KAGGLE_PREFIX + x)


# combine the dataframes
combined_data = pd.concat([our_data, kaggle_data], ignore_index=True)
# shuffle the data
combined_data = combined_data.sample(frac=1).reset_index(drop=True)
# save the data
combined_data.to_csv("../../data/combined_data.csv", index=False, header=False)
