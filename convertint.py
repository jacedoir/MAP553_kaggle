import pandas as pd
import numpy as np

data = pd.read_csv('predictions_tpot_splited_pipeline.csv', sep=',', dtype=np.float64)

data['Id'] = data['Id'].astype(int)
data['Cover_type'] = data['Cover_type'].astype(int)

data.to_csv('predictions_tpot_splited_pipeline_int.csv', index=False)