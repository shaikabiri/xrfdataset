import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets, linear_model
from sklearn.cross_decomposition import PLSRegression
from scipy import stats
from os import cpu_count
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
import warnings
from sklearn import linear_model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, Matern, ConstantKernel, Product
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
from sklearn.ensemble import IsolationForest
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPRegressor
import scipy.stats as stats
import composition_stats as cs
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.ensemble import GradientBoostingRegressor

def core_processor(excel_file_address,depth_int,columns,int_size,sav_par):
    raw = pd.read_excel(excel_file_address, sheet_name=0,skiprows=0)
    raw.dropna(inplace=True)
    raw.reset_index(inplace=True,drop=True)
    raw['PS position (cm)'] = raw['PS position (cm)'] - raw['PS position (cm)'][0]
    raw['PS position (cm)'] = raw['PS position (cm)']/2


    raw.drop(labels=raw.index[raw['sample surface']>7].tolist(),axis=0,inplace=True)
    #load elements
    Xcore = raw[columns]

    #replace zeroes
    
    Xcore = Xcore.div(raw['kcps'], axis=0)
    for i in Xcore.columns:
        Xcore.loc[Xcore[i] == 0, i] = min(Xcore.loc[Xcore[i] != 0, i])*(1/3)
    

    # #find anomalies in dataset and set them to median
    iso = IsolationForest(random_state=0)
    yhat = iso.fit_predict(Xcore)
    mask = yhat == -1
    Xcore.iloc[mask, :] = np.nan
    names = Xcore.columns
    imp_mean = IterativeImputer(random_state=0)
    Xcore = imp_mean.fit_transform(Xcore)
    Xcore = pd.DataFrame(Xcore)
    Xcore.columns = names
    # #remove noise 
    for i in Xcore.columns:
        Xcore[i] = savgol_filter(Xcore[i],int(sav_par.loc[i,'width']),2)
    Depths = raw['PS position (cm)']
    Depths = Depths.to_numpy()
    X_core_10cm_mean = pd.DataFrame(columns=Xcore.columns)

    for i in range(depth_int):
        a = Depths<(i+1)*int_size
        b = Depths>=i*int_size
        X_core_10cm_mean.loc[i] = Xcore.loc[a==b,:].median()

    return X_core_10cm_mean


