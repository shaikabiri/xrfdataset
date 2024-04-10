#logs
"""  
---------------------------------------------------------------------------------------------------
3/1/2024:
 - I think we are done for the most parts
---------------------------------------------------------------------------------------------------
"""
###################################################################################################
#import packages
from scipy.signal import savgol_filter, general_gaussian
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
from helpers import core_processor
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFE
import pywavelet as pywt
from skimage.restoration import denoise_wavelet
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import re
from sklearn import preprocessing as pre
from scipy.stats.mstats import gmean
from pykrige.rk import RegressionKriging
from sklearn.gaussian_process.kernels import (RationalQuadratic,  Exponentiation)
from scipy import stats
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
from statsmodels.stats import diagnostic as diag
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor 
##################################################################################################
####calculate percent deviations for the reference data###########################################
##################################################################################################

#load reference data and repeats
ref10CM = pd.read_excel('ref.xlsx', sheet_name=0,skiprows=0)
ref1CM = pd.read_excel('ref.xlsx', sheet_name=1,skiprows=0)
reps10CM = pd.read_excel('ref.xlsx', sheet_name= 2)
reps1CM = pd.read_excel('ref.xlsx', sheet_name= 3)

#get the units for each element
units = ref10CM.loc[0,ref10CM.iloc[0,:].notna()]

#get the limits of detection for each element
limit_detection = ref10CM.loc[len(ref10CM)-1,ref10CM.iloc[len(ref10CM)-1,:].notna()]
limit_detection.drop(labels='Sample.no',inplace=True)

#drop the labels from reference and repeats and reset indices
ref10CM.drop(labels=[0,len(ref10CM)-1],axis=0,inplace=True)
ref1CM.drop(labels=[0,len(ref1CM)-1],axis=0,inplace=True)
reps10CM.drop(labels=[0,len(reps10CM)-1],axis=0,inplace=True)
reps1CM.drop(labels=[0,len(reps1CM)-1],axis=0,inplace=True)

reps10CM.reset_index(inplace=True)
reps1CM.reset_index(inplace=True)
ref1CM.reset_index(inplace=True)
ref10CM.reset_index(inplace=True)

#save core labels
cores = ref10CM['Soil.core.ID']

#calculate means and stds for 1cm and calculate CV
mean_reps1CM = reps1CM.groupby(['X1.cm.depth'])[reps1CM.columns[5:]].mean()
std_reps1CM = reps1CM.groupby(['X1.cm.depth'])[reps1CM.columns[5:]].std()
CV_1cm = (std_reps1CM/mean_reps1CM)*100
CV_1cm = CV_1cm.mean()

#plot error in %CV for 1cm
ax = sns.barplot(x=CV_1cm.index, y=CV_1cm.values, color='grey')
ax.set(xlabel='Elements', ylabel='% CV for 1cm')
plt.show()

#convert rep ids to string for 10cm
reps10CM['Rep.ID'] = reps10CM['Rep.ID'].astype(str).str[0]

#calculate means and stds for 10cm and calculate CV
mean_reps10CM = reps10CM.groupby(['Rep.ID'])[reps10CM.columns[5:]].mean()
std_reps10CM = reps10CM.groupby(['Rep.ID'])[reps10CM.columns[5:]].std()
CV_10cm = (std_reps10CM/mean_reps10CM)*100
CV_10cm = CV_10cm.mean()

#calculate raw errors for each element in 10cm
errors = (CV_10cm/100)*mean_reps10CM.mean()

#plot the errors in %CV for 10cm
ax = sns.barplot(x=CV_10cm.index, y=CV_10cm.values, color='grey')
ax.set(xlabel='Elements', ylabel='% CV for 10cm')
plt.show()


##################################################################################################
####import pellet data############################################################################
##################################################################################################

folder = 'Pellets'
#get all the pellet folders
sub_folders = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]

#get the means and stds of the pellets
pellet_means = pd.DataFrame(columns=['cps', 'Al', 'Si', 'S', 'Cl',
       'Ar', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As',
       'Br', 'Rb', 'Sr', 'Zr', 'Sb', 'Ba', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd'])
pellet_std = pd.DataFrame(columns=['cps', 'Al', 'Si', 'S', 'Cl',
       'Ar', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As',
       'Br', 'Rb', 'Sr', 'Zr', 'Sb', 'Ba', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd'])

for folder in sub_folders:
    data = pd.read_csv('Pellets/' + folder + "/result.txt",sep='\t',header=2)
    pellet_means.loc[len(pellet_means)] = data[['cps', 'Al', 'Si', 'S', 'Cl',
       'Ar', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As',
       'Br', 'Rb', 'Sr', 'Zr', 'Sb', 'Ba', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd']].mean()
    pellet_std.loc[len(pellet_std)] = data[['cps', 'Al', 'Si', 'S', 'Cl',
       'Ar', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As',
       'Br', 'Rb', 'Sr', 'Zr', 'Sb', 'Ba', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd']].std()
   
pellets_depth =  pd.read_excel('pellets_depth.xlsx')
#divide pellet counts by cps and drop it
pellet = pellet_means.divide(pellet_means['cps'],axis=0)
pellet = pellet.drop('cps',axis=1)

#calculate CV values for pellets
CV = pellet_std/pellet_means
CVmean = CV.mean().sort_values()*100

##################################################################################################
####establish Savitsky-Golay intervals############################################################
##################################################################################################


#load BE15 raw core
BE15 = pd.read_excel('Cores/BE15.xlsx', sheet_name=0,skiprows=0)
BE15.dropna(inplace=True)
BE15.reset_index(inplace=True,drop=True)
BE15['PS position (cm)'] = BE15['PS position (cm)'] - BE15['PS position (cm)'][0]
BE15['PS position (cm)'] = BE15['PS position (cm)']/2
BE15.drop(labels=BE15.index[BE15['sample surface']>7].tolist(),axis=0,inplace=True)
#load elements
Xcore = BE15[['Mg', 'Al', 'Si', 'P', 'S', 'Cl',
       'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',
       'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Rb', 'Sr', 'Y', 'Zr', 'Cd', 'Sn',
       'Ba', 'La', 'Ce', 'Ta', 'W', 'Pb', 'Bi', 'U']]

#divide by total counts
Xcore = Xcore.div(BE15['kcps'], axis=0)

#define outputs of this section
element_sav_par = pd.DataFrame({'elements' : [],'width' : []})
element_sav_par['elements'] = Xcore.columns
element_sav_par.set_index('elements',drop=True,inplace=True)

#replace zeroes
for i in Xcore.columns:
    Xcore.loc[Xcore[i] == 0, i] = min(Xcore.loc[Xcore[i] != 0, i])*(1/3)

#find anomalies in dataset and set them to median
iso = IsolationForest(random_state=0)
yhat = iso.fit_predict(Xcore)
mask = yhat == -1
Xcore.iloc[mask, :] = np.nan
names = Xcore.columns
imp_mean = IterativeImputer(random_state=0)
Xcore = imp_mean.fit_transform(Xcore)
Xcore = pd.DataFrame(Xcore)
Xcore.columns = names
#save Ca before denoise
Ca_before_denoise = Xcore['Ca']

#if params are not already save run this loop
for elem in Xcore.columns:
    X = Xcore[elem]
    # Calculate the power spectrum 
    ps1 = np.abs(np.fft.fftshift(np.fft.fft(X)))**2

    # Define pixel in original signal and Fourier Transform
    pix1 = BE15['PS position (cm)']
    fpix1 = BE15['PS position (cm)'] - np.max(BE15['PS position (cm)'])/2

    with plt.style.context(('ggplot')):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        fig.text(0.5, 0.04, 'Depth (cm)', ha='center')
        fig.text(0.04, 0.5, 'Ca counts', va='center', rotation='vertical')
        axes[0].plot(pix1, X)
        axes[0].set_title('A', fontsize='small', loc='left',fontweight='bold',size=12)
        axes[1].semilogy(fpix1, ps1, 'b')
        axes[1].set_title('B', fontsize='small', loc='left',fontweight='bold',size=12)

    # Determine the smooth section of the function
    plt.show()
    st = float(input('What is the start of smooth section?\n'))
    en = float(input('What is the end of smooth section?\n'))
    start = BE15.index[np.logical_and(BE15['PS position (cm)']<(st+.01),BE15['PS position (cm)']>(st-0.01))].tolist()[0]
    end = BE15.index[np.logical_and(BE15['PS position (cm)']<(en+.01),BE15['PS position (cm)']>(en-0.01))].tolist()[0]

    # Calculate the power spectrum 
    ps = np.abs(np.fft.fftshift(np.fft.fft(X[start:end])))**2
    
    # Define pixel in original signal and Fourier Transform
    pix = BE15['PS position (cm)'][start:end]
    fpix = pix - (np.max(pix))/2 -2.5

    with plt.style.context(('ggplot')):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
        fig.text(0.5, 0.04, 'Depth (cm)', ha='center')
        fig.text(0.04, 0.5, 'Calcium counts', va='center', rotation='vertical')
        axes[0,0].plot(pix1, X)
        axes[0,0].set_title('A', fontsize='small', loc='left',fontweight='bold',size=12)
        axes[0,1].semilogy(fpix1, ps1, 'b')
        axes[0,1].set_title('C', fontsize='small', loc='left',fontweight='bold',size=12)

    with plt.style.context(('ggplot')):
        fig.text(0.5, 0.5, 'Power spectrum', va='center', rotation='vertical')
        axes[1,0].plot(pix, X[start:end])
        axes[1,0].set_title('B', fontsize='small', loc='left',fontweight='bold',size=12)
        axes[1,1].semilogy(fpix, ps, 'b')
        axes[1,1].set_title('D', fontsize='small', loc='left',fontweight='bold',size=12)

    plt.savefig('Ca_before_noise_removal.png',dpi=300,pad_inches=0.5,bbox_inches='')
    plt.show()

    # Set some reasonable parameters to start with

    w = [101,201,401,501,801]
    # Define pixel in Fourier space
    i=0
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(12, 3))
    fig.tight_layout(pad=2)
    with plt.style.context(('ggplot')):
        for widths in w:
            X_smooth = savgol_filter(X, widths, polyorder = 2)
            ps = np.abs(np.fft.fftshift(np.fft.fft(X_smooth[start:end])))**2    
            axes[i].semilogy(fpix, ps, 'b')
            axes[i].tick_params(axis='both', which='major', labelsize=7)
            axes[i].set_title('Width: '+str(widths/50) +'cm',{'fontsize':10})
            i = i+1
    axes[0].set_ylabel('Power spectrum')
    plt.savefig('Fouriers_for_Ws.png',dpi=300,pad_inches=2.5)
    plt.show()
    element_sav_par.loc[elem,'width'] = float(input('What is the width of this element?\n'))

element_sav_par.to_csv('elements_sav_par1.csv')

#if params already saved run this
element_sav_par = pd.read_csv('elements_sav_par.csv')
element_sav_par.set_index('elements',drop=True,inplace=True)
##################################################################################################
####get all the core data#########################################################################    
##################################################################################################

#elements needed
Xcols = ['Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V',
       'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
       'Rb', 'Sr', 'Y', 'Zr', 'Cd', 'Sn', 'Ba', 'La', 'Ce', 'Ta', 'W', 'Pb',
       'Bi', 'U']
Ycols = ['Sc', 'V', 'Cr', 'Mn', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Rb',
       'Zr', 'Sn', 'Y', 'Ba', 'La', 'Ce', 'Cd', 'Pb', 'Bi', 'U', 'Se', 'Al',
       'Ca', 'Fe', 'K', 'Mg', 'P', 'S', 'Sr']

#empty data frames for data
tdata = pd.DataFrame(columns=pd.concat([pd.Series(Xcols),pd.Series("coreid")]))
ydata = pd.DataFrame(columns=pd.concat([pd.Series(Ycols),pd.Series("coreid")]))

#address of core files
core_files = os.listdir('Cores')
#address of reference file
ref10CM = pd.read_excel('ref.xlsx', sheet_name=0,skiprows=0)
#load reference
coreids = ref10CM['Soil.core.ID'].dropna().unique()
#number of intervals on each core in order
intervals = [9,9,9,10,10,10,9,10,10]

#loop that load core files
for k in range(9):
    #core_processor does the Isolation Forest and Savitsky-Golay and median averaging, takes address, number of intervals, columns of X to be used and interval size
    cr = core_processor("Cores/"+core_files[k],intervals[k],Xcols,10,element_sav_par)
    #add core id
    cr['coreid'] = core_files[k].replace('.xlsx','')
    #append to X data
    tdata = pd.concat([tdata,cr])
    #read reference 
    ref10CM = pd.read_excel('ref.xlsx', sheet_name=0,skiprows=0)
    ref10CM = ref10CM.loc[ref10CM['Soil.core.ID']==coreids[k],:]
    #there is an additional reference point on core 3.7P not available on scanned cores
    if coreids[k] == '3.7P':
        ref10CM = ref10CM.iloc[0:10,:]
    #reset index
    ref10CM.reset_index(drop=True,inplace=True)
    #append y
    y = ref10CM[Ycols] 
    y['coreid'] = core_files[k].replace('.xlsx','')
    ydata = pd.concat([ydata,y])
#reset index
ydata = ydata.reset_index(drop=True)
tdata.reset_index(inplace=True,drop=True)

#plot and save correlations for the reference data
plt.figure(figsize=(7.5,9))
corr = ydata.drop('coreid',axis=1).sort_index(axis=1).corr()
mat = np.triu(corr,k=1)
sns.heatmap(round(corr,1), cmap="Blues", annot=True,mask=mat,xticklabels=True,yticklabels=True,annot_kws={"size": 7,'rotation':'vertical'})
plt.xticks(fontsize=8,rotation='vertical')
plt.yticks(fontsize=8)
plt.savefig('conc_corr.png',dpi=300,pad_inches=0.0)
plt.show()

#plot relative conc. of cores
fig, axes = plt.subplots(6,6,sharex=False,sharey=True,figsize=(8, 12))

p=0
for j in range(6):
    for i in range(6):   
        elem = ydata.drop('coreid',axis=1).columns.sort_values()[p]
        sns.boxplot(ax=axes[j,i], data=ref10CM, x=elem, y=cores)
        axes[j,i].set_xlabel(elem+ " (" + units[elem] + ")",fontsize=10)
        axes[j,i].set_ylabel("")
        axes[j,i].yaxis.set_tick_params(labelsize=8)
        axes[j,i].xaxis.set_tick_params(labelsize=8)
        p = p+1

plt.subplots_adjust(wspace=0.1, 
                    hspace=0.5)
plt.savefig('desc_conc.png',dpi=300,pad_inches=0.0)

#remove the rows not in the pellet dataset
core_for_pellet = tdata.drop(tdata[np.logical_or(np.logical_or(tdata.loc[:,'coreid']=="BE15", tdata.loc[:,'coreid']=="G25"),tdata.loc[:,'coreid']=="P31")].index)
ref_for_pellet = ydata.drop(tdata[np.logical_or(np.logical_or(tdata.loc[:,'coreid']=="BE15", tdata.loc[:,'coreid']=="G25"),tdata.loc[:,'coreid']=="P31")].index)
core_for_pellet.reset_index(drop=True,inplace=True)
ref_for_pellet.reset_index(drop=True,inplace=True)
core_for_pellet = core_for_pellet.drop([17,34, 35, 36, 37])
ref_for_pellet = ref_for_pellet.drop([17,34, 35, 36, 37])
core_for_pellet.reset_index(drop=True,inplace=True)
ref_for_pellet.reset_index(drop=True,inplace=True)

#find common elements between cores and pellets
common_elements = np.intersect1d(core_for_pellet.columns, pellet_means.columns)

#get correlation values between cores and pellets
corrs_raw = core_for_pellet.corrwith(pellet).sort_values().dropna()

#save descriptives of core counts to csv file
counts_desk = pd.DataFrame(columns=['mean','std','first_quant','third_quant','%CV','r'],index=common_elements)
counts_desk['mean'] = tdata[common_elements].mean()
counts_desk['std'] = tdata[common_elements].std()
counts_desk['first_quant'] = tdata[common_elements].quantile(0.25)
counts_desk['third_quant'] = tdata[common_elements].quantile(0.75)
counts_desk['%CV'] = CVmean[common_elements]
counts_desk['r'] = corrs_raw[common_elements]
counts_desk.to_csv('counts_desc.csv')
#correlation between the ys and xs
ydata.corrwith(tdata[common_elements]).loc[ydata.columns].sort_index().to_csv('y_corr_x.csv')
#plot correlation plot for core counts
plt.figure(figsize=(7.5*(22/31),9*(22/31)))
corr = tdata[common_elements].sort_index(axis=1).corr()
mat = np.triu(corr,k=1)
sns.heatmap(round(corr,1), cmap="Blues", annot=True,mask=mat,xticklabels=True,yticklabels=True,annot_kws={"size": 7,'rotation':'vertical'})
plt.xticks(fontsize=8,rotation='vertical')
plt.yticks(fontsize=8)
plt.savefig('counts_corr.png',dpi=300,pad_inches=0.0)
plt.show()

#plot correlation coefficients between pellets and cores
plt.figure(figsize=(7,5))
graph = sns.barplot(x=corrs_raw.index,y=corrs_raw,facecolor=(0.5, 0.5, 0.5, 1.0))
#Drawing a horizontal line at point 1.25
graph.axhline(0,color=(0,0,0,1))
graph.axhline(0.6,color=(0,0,0,1),linestyle='--')
plt.yticks(fontsize=6)
plt.xticks(fontsize=6)
plt.ylabel("Correlation coefficient between pellets and\n cores element counts",fontsize=10)
plt.savefig('pellet_corr_correl_raw.png',dpi=300,pad_inches=0.0)
plt.show()

#drop elements that are below 0.6
correlative_elements = corrs_raw[corrs_raw>0.6].index.to_list()
correlative_tdata = tdata[correlative_elements]

# VIF dataframe 
vif_data = pd.DataFrame() 
vif_data["feature"] = correlative_tdata.columns 
  
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(correlative_tdata.values, i) 
                          for i in range(len(correlative_tdata.columns))] 
  
vif_data = vif_data.sort_values(by="VIF").reset_index(drop=True)
features = vif_data.sort_values(by="VIF").reset_index(drop=True)[0:7]['feature']

#plot VIF
plt.figure(figsize=(7,5))
graph = sns.barplot(x=vif_data["feature"],y=vif_data["VIF"],facecolor=(0.5, 0.5, 0.5, 1.0))
#Drawing a horizontal line at point 1.25
graph.axhline(100,color=(0,0,0,1),linestyle='--')
plt.yticks(fontsize=6)
plt.xticks(fontsize=6)
plt.ylabel("Variance inflation factor",fontsize=10)
plt.savefig('VIF.png',dpi=300,pad_inches=0.0)
plt.show()

reduced_X = correlative_tdata[features]

###correlation plots for reduced X
#plot correlation plot for core counts
plt.figure(figsize=(7.5*(22/31),9*(22/31)))
corr = reduced_X.sort_index(axis=1).corr()
mat = np.triu(corr,k=1)
sns.heatmap(round(corr,1), cmap="Blues", annot=True,mask=mat,xticklabels=True,yticklabels=True,annot_kws={"size": 7,'rotation':'vertical'})
plt.xticks(fontsize=8,rotation='vertical')
plt.yticks(fontsize=8)
plt.savefig('reducedX_corr.png',dpi=300,pad_inches=0.0)
plt.show()


####################################################################################################################
#Plot PCAs############################################################################################
####################################################################################################################


#scale and perform PCA for core counts
pca = PCA()
scl = StandardScaler()
X_pca = scl.fit_transform(tdata.iloc[:,0:38])
X_pca = pca.fit_transform(X_pca)

#calculate explained variance
var = pca.explained_variance_
var = var/np.sum(var)*100

#plot variance explaned and PC1,PC2 and PC3
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(1,2,1)

sns.barplot(x=[1,2,3,4,5,6,7,8,9,10],y=var[0:10], color = (254/255,178/255,76/255,1),ax=ax1)
ax1.set_xlabel("PC num.")
ax1.set_ylabel("%Var explained")
tdata['Soil type'] = 'type'
for i in range(len(tdata)):
    if 'BE' in (tdata['coreid'][i]):
        tdata['Soil type'][i] = "BE"
    if 'G' in (tdata['coreid'][i]):
        tdata['Soil type'][i] = "G"   
    if 'P' in (tdata['coreid'][i]):
        tdata['Soil type'][i] = "P"


pal = ['#1b9e77', '#d95f02', '#7570b3']


colors = {'BE':pal[0], 'P':pal[1], 'G':pal[2]}
import matplotlib.patches as mpatches
patch1 = mpatches.Circle((0,0),1,color=pal[0], label='BE')
patch2 = mpatches.Circle((0,0),1,color=pal[1], label='P')
patch3 = mpatches.Circle((0,0),1,color=pal[2], label='G')

ax2 = fig.add_subplot(1,2,2, projection='3d')
ax2.scatter(X_pca[:,0],X_pca[:,1],X_pca[:,2], c=tdata['Soil type'].map(colors),alpha=1)
ax2.set_xlabel("PC1 (" + str(round(var[0],1))+"%)")
ax2.set_ylabel("PC2 (" + str(round(var[1],1))+"%)")
ax2.set_zlabel("PC3 (" + str(round(var[2],1))+"%)")
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False
handles, labels = ax2.get_legend_handles_labels()
handles.append(patch1)
handles.append(patch2) 
handles.append(patch3) 
ax2.margins(x=0,y=0)
ax1.set_title('A',loc='left')
ax2.set_title('B',loc='left',pad=15)



plt.legend(handles=handles, loc="upper left")
plt.savefig('xpca.png',dpi=300,pad_inches=0.5,bbox_inches='')
plt.show()


###correlation plots for xPCA
#plot correlation plot for core counts
plt.figure(figsize=(7.5*(22/31),9*(22/31)))
X_pca = pd.DataFrame(X_pca).sort_index(axis=1)
X_pca.columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7']
corr = X_pca.corr()
mat = np.triu(corr,k=1)
sns.heatmap(round(corr,1), cmap="Blues", annot=True,mask=mat,xticklabels=True,yticklabels=True,annot_kws={"size": 7,'rotation':'vertical'})
plt.xticks(fontsize=8,rotation='vertical')
plt.yticks(fontsize=8)
plt.savefig('xpca_corr.png',dpi=300,pad_inches=0.0)
plt.show()

###correlation plots for xPCA and responses
#plot correlation plot for core counts
plt.figure(figsize=(6,8))
Y = ydata.drop(['coreid'],axis=1).copy().sort_index(axis=1)
corr = pd.concat([Y, X_pca], axis=1, keys=['Y', 'X_pca']).corr().loc['Y', 'X_pca']
sns.heatmap(round(corr,1), cmap="Blues", annot=True,xticklabels=True,yticklabels=True,annot_kws={"size": 7,'rotation':'horizontal'})
plt.xticks(fontsize=8,rotation='vertical')
plt.yticks(fontsize=8)
plt.savefig('pcaXvsResponse_corr.png',dpi=300,pad_inches=0.0)
plt.show()



###pca for ys
#scale and perform PCA for reference
pca = PCA()
scl = StandardScaler()
X_pca = scl.fit_transform(ydata.drop('coreid',axis=1))
X_pca = pca.fit_transform(X_pca)


var = pca.explained_variance_
var = var/np.sum(var)*100
fig = plt.figure(figsize=(8,4))

ax1 = fig.add_subplot(1,2,1)

sns.barplot(x=[1,2,3,4,5,6,7,8,9,10],y=var[0:10], color = (254/255,178/255,76/255,1),ax=ax1)
ax1.set_xlabel("PC num.")
ax1.set_ylabel("%Var explained")
tdata['Soil type'] = 'type'
for i in range(len(tdata)):
    if 'BE' in (tdata['coreid'][i]):
        tdata['Soil type'][i] = "BE"
    if 'G' in (tdata['coreid'][i]):
        tdata['Soil type'][i] = "G"   
    if 'P' in (tdata['coreid'][i]):
        tdata['Soil type'][i] = "P"


pal = ['#1b9e77', '#d95f02', '#7570b3']


colors = {'BE':pal[0], 'P':pal[1], 'G':pal[2]}
import matplotlib.patches as mpatches
patch1 = mpatches.Circle((0,0),1,color=pal[0], label='BE')
patch2 = mpatches.Circle((0,0),1,color=pal[1], label='P')
patch3 = mpatches.Circle((0,0),1,color=pal[2], label='G')

ax2 = fig.add_subplot(1,2,2, projection='3d')
ax2.scatter(X_pca[:,0],X_pca[:,1],X_pca[:,2], c=tdata['Soil type'].map(colors),alpha=1)
ax2.set_xlabel("PC1 (" + str(round(var[0],1))+"%)")
ax2.set_ylabel("PC2 (" + str(round(var[1],1))+"%)")
ax2.set_zlabel("PC3 (" + str(round(var[2],1))+"%)")
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False
handles, labels = ax2.get_legend_handles_labels()
handles.append(patch1)
handles.append(patch2) 
handles.append(patch3) 
ax2.margins(x=0,y=0)
ax1.set_title('A',loc='left')
ax2.set_title('B',loc='left',pad=15)



plt.legend(handles=handles, loc="upper left")
plt.savefig('pca_conc.png',dpi=300,pad_inches=0.5,bbox_inches='')
plt.show()





####################################################################################################################
#single core calibration############################################################################################
####################################################################################################################
#remove noise 
for i in Xcore.columns:
    Xcore[i] = savgol_filter(Xcore[i],int(element_sav_par.loc[i,'width']),2)
Depths = BE15['PS position (cm)']
Depths = Depths.to_numpy()

#plot Ca before and after noise removal
Ca_after_denoise = Xcore['Ca']

#plot Ca before and after denoise
with plt.style.context(('ggplot')):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    fig.text(0.5, 0.04, 'Depth (cm)', ha='center')
    fig.text(0.04, 0.5, 'Calcium counts', va='center', rotation='vertical')
    axes[0].plot(Depths, Ca_before_denoise)
    axes[0].set_title('A', fontsize='small', loc='left',fontweight='bold',size=12)
    axes[1].plot(Depths, Ca_after_denoise)
    axes[1].set_title('B', fontsize='small', loc='left',fontweight='bold',size=12)

plt.savefig('Ca_before_after.png',dpi=300,pad_inches=0.5,bbox_inches='')

plt.show()

#create average X for cores
X_core_10cm_mean = pd.DataFrame(columns=Xcore.columns)
X_core_1cm_mean = pd.DataFrame(columns=Xcore.columns)
Moinc10 = pd.Series()
Mocoh10 = pd.Series()

# for 10 cm
for i in range(9):
    a = Depths<(i+1)*10
    b = Depths>=i*10
    X_core_10cm_mean.loc[i] = Xcore.loc[a==b,:].median()
    Moinc10[i] = BE15.loc[a==b,'Mo inc'].median()
    Mocoh10[i] = BE15.loc[a==b,'Mo coh'].median()

Moinc1 = pd.Series()
Mocoh1 = pd.Series()

# for 1 cm
for i in range(86):
    a = Depths<(i+1)
    b = Depths>=i
    X_core_1cm_mean.loc[i] = Xcore.loc[a==b,:].median()
    Moinc1[i] = BE15.loc[a==b,'Mo inc'].median()
    Mocoh1[i] = BE15.loc[a==b,'Mo coh'].median()


X_core_1cm_mean.reset_index(inplace=True)
X_core_1cm_mean['Depth'] = (X_core_1cm_mean['index'] + 0.5)
Xtrain = X_core_10cm_mean

#load references
ref1CM = pd.read_excel('ref.xlsx', sheet_name=1,skiprows=0)
ref1CM.drop(labels=[0,len(ref1CM)-1],axis=0,inplace=True)
ref1CM.reset_index(drop=True,inplace=True)

Xtest = pd.DataFrame(columns=Xtrain.columns)
q = 0
for i in range(len(X_core_1cm_mean)):
    if (X_core_1cm_mean['Depth'][i] in ref1CM['X1cm.depth'].to_numpy()):
        Xtemp = X_core_1cm_mean[Xtest.columns]
        Xtest.loc[q] = Xtemp.iloc[i,:]
        q = q+1

ref10CM = pd.read_excel('ref.xlsx', sheet_name=0,skiprows=0)
units = ref10CM.loc[0,ref10CM.iloc[0,:].notna()]
ref10CM = ref10CM.loc[ref10CM['Soil.core.ID']=='BE2',:]
ref10CM.reset_index(drop=True,inplace=True)

Ytest = ref1CM.iloc[:,3:]


Ytrain = ref10CM.iloc[:,3:]


Ytrain = Ytrain.astype(float)
Ytest = Ytest.astype(float)

#reduce the dataset
Xtrain = Xtrain[features]
Xtest = Xtest[features]
 
#model for univariate and 1cm as validation set without krieging 

res2 = pd.DataFrame(columns=['Element','R2','RMSE','denom'])
for j in Ytrain.columns:
    res = []
    r = 0
    rmse2 = 0
    for i in range(1,8):
        x = (Xtrain.to_numpy())
        y = (Ytrain[j].to_numpy())
        mdl = PLSRegression(i)
        mdl.fit(x, y)

        Xt = (Xtest.to_numpy())

        pred3= mdl.predict(Xt)
        pred3 = (pred3)
        #pred3 = savgol_filter(pred3.reshape(1,-1),int(element_sav_par.loc[j,'width']/50)+1,2)
        #pred3 = pred3[0]
        if (pred3<0).any():
            pred3[pred3<0] = np.min(limit_detection[j])
        r2 = r2_score((Ytest[j]),(pred3))
        rmse = np.sqrt(mean_squared_error(Ytest[j].to_numpy().reshape(-1,1),((pred3).reshape(-1,1))))
        cv = rmse/np.mean(Ytest[j].to_numpy().reshape(-1,1))
        if r2 > r:
            r = r2
            rmse2 = rmse
            res = i
    res2.loc[len(res2)] = [j,r,rmse2,res]
res_univariate_15BE_linear = res2.sort_values('R2',ascending=False)
res_univariate_15BE_linear.reset_index(inplace=True,drop=True)
res_univariate_15BE_linear.set_index('Element',inplace=True)
res_univariate_15BE_linear.sort_index().to_csv('single_core_without_nopr.csv')


#create plots of the best 5 R2 elements
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
fig.tight_layout(w_pad=5,h_pad=2.5,pad=2)
elements = ['Ca','P','S']
titles = ['A','D','B','G','H','I']
i=0
for elem in elements:
    denom = res_univariate_15BE_linear.loc[elem,'denom']
    R2 = res_univariate_15BE_linear.loc[elem,'R2']
    RMSE = res_univariate_15BE_linear.loc[elem,'RMSE']
    x = (Xtrain.to_numpy())
    y = (Ytrain[elem].to_numpy())

    mdl = PLSRegression(denom)

    
    mdl.fit(x, y)

    Xt = (Xtest.to_numpy())

    pred3= mdl.predict(Xt)
    pred3 = (pred3)
    
    pred3 = savgol_filter(pred3.reshape(1,-1),int(element_sav_par.loc[elem,'width']/50)+1,2)

    pred3 = pred3[0]
    if (pred3<0).any():
        pred3[pred3<0] = limit_detection[elem]

    sns.lineplot(x=ref10CM['X10cm.depth'],y=Ytrain[elem],ax=axes[0,i],color=(49/255,130/255,189/255,1))
    sns.lineplot(x=ref10CM['X10cm.depth'], y= Ytrain[elem]+errors[elem],linestyle='dashed',ax=axes[0,i], color=(49/255,130/255,189/255,1))
    sns.lineplot(x=ref10CM['X10cm.depth'], y= Ytrain[elem]-errors[elem],linestyle='dashed',ax=axes[0,i], color=(49/255,130/255,189/255,1))
    mx = max([max(Ytrain[elem].to_numpy()),max(Ytest[elem].to_numpy()),np.max(pred3)])
    axes[0,i].set_ylabel('')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    mx = 0.1*mx + mx 
    axes[0,i].set_ylim(0,mx)
    axes[0,i].set_xlabel("")
    axes[0,i].set_xlim(0,90)
    axes[0,i].set_title(titles[i*2], fontsize='small', loc='left',fontweight='bold',size=12)
    #axes[0,i].set_title(elem + " ("+units[elem] + ")",fontsize=15)
    axes[0,i].tick_params(axis='x', labelsize=15)
    axes[0,i].tick_params(axis='y', labelsize=15)
    sns.lineplot(x= ref1CM['X1cm.depth'],y=Ytest[elem],ax=axes[1,i],linestyle="--",color=(0.5,0.5,0.5,1))
    axes[1,i].fill_between(ref1CM['X1cm.depth'], Ytest[elem]-errors[elem], Ytest[elem]+errors[elem], alpha=0.2)
    sns.lineplot(x= ref1CM['X1cm.depth'],y=pred3,ax=axes[1,i],color=(252/255,141/255,98/255,1))
    #sns.lineplot(x= ref1CM['X1cm.depth'],y=pred3+RMSE*1.96,ax=ax2,color=(252/255,141/255,98/255,1),linestyle='dashed')
    #ns.lineplot(x= ref1CM['X1cm.depth'],y=pred3-RMSE*1.96,ax=ax2,color=(252/255,141/255,98/255,1),linestyle='dashed')
    axes[1,i].tick_params(axis='x', labelsize=15)
    axes[1,i].tick_params(axis='y', labelsize=15)
    axes[1,i].set_ylim(0,mx)
    axes[1,i].set_xlim(0,90)
    axes[1,i].set_xlabel("")
    axes[1,i].set_ylabel("")
    axes[1,i].set_title(titles[i*2+1], fontsize='small', loc='left',fontweight='bold',size=12)
    R2text = ('$R^2$ = '+ str(round(R2,2)) + "\n" + "$RMSE$ =" + str(round(RMSE,2)))
    axes[1,i].text(x=5,y=(mx-mx*0.2),s=R2text,fontsize=15)
    i = i+1

fig.text(0.01, 0.5, 'Ca (%)', va='center', rotation='vertical',fontsize=14)
fig.text(0.32, 0.5, 'P (%)', va='center', rotation='vertical',fontsize=14)
fig.text(0.65, 0.5, 'S (%)', va='center', rotation='vertical',fontsize=14)
plt.savefig('predicted_elems_woa.png',dpi=300,pad_inches=0.0)
plt.show()




##################################################################################################################
######complete dataset calibration################################################################################
##################################################################################################################

#load X data
xdata = tdata[features]

#multivariate model for all the dataset
res2 = pd.DataFrame(columns=['R2','RMSE','Predictor'],index=Ycols)

for i in Ycols:
    r = 0
    for k in range(1,8):
        ypred_acum = np.empty(0)
        ytest_acum = np.empty(0)
        xtest_acum = np.empty(0)

        for j in pd.unique(tdata['coreid']):
            xtest = xdata.loc[tdata['coreid']==j,:]
            ytest = (ydata.loc[ydata['coreid']==j,i].to_numpy())
            xtrain = xdata.loc[np.logical_not(tdata['coreid']==j),:]
            ytrain = (ydata.loc[np.logical_not(ydata['coreid']==j),i].to_numpy())
            mdl = PLSRegression(k)
            mdl.fit(xtrain,ytrain)
            pred = mdl.predict(xtest)
            ypred_acum = np.append(ypred_acum,(pred))
            ytest_acum = np.append(ytest_acum,(ytest))
            xtest_acum = np.append(xtest_acum,xtest)

        r2 = r2_score(ytest_acum,ypred_acum)
        rmse = np.sqrt(mean_squared_error(ytest_acum,ypred_acum))
        if r2 > r:
            r = r2
            rmse2 = rmse
            res = k
    res2.loc[i] = [r,rmse2,res]

            
res_univariate_all_linear = res2
res_univariate_all_linear.sort_values(by='R2')


#create plots of the best 5 R2 elements
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
fig.tight_layout(w_pad=5,h_pad=2.5,pad=4)
elements = ['Ca','Zn', 'Pb']
cores = ['BE15','G23','P35']
# titles = ['A','D','B','E','F','G']

m=0

for elem in elements:
    i=0
    for p in cores:
        denom = res_univariate_all_linear.loc[elem,'Predictor']
        R2 = res_univariate_all_linear.loc[elem,'R2']
        RMSE = res_univariate_all_linear.loc[elem,'RMSE']
        x = xdata.loc[np.logical_not(tdata['coreid']==p),:]
        y = (ydata.loc[np.logical_not(ydata['coreid']==p),elem].to_numpy())

        mdl = PLSRegression(denom)

        
        mdl.fit(x, y)

        Xt = xdata.loc[tdata['coreid']==p,:]
        Yt = (ydata.loc[ydata['coreid']==p,elem].to_numpy())

        pred3= mdl.predict(Xt)
        
        # pred3 = savgol_filter(pred3.reshape(1,-1),int(element_sav_par.loc[elem,'width']/50)+1,2)
        pred3= pred3.reshape(-1,)
        if (pred3<0).any():
            pred3[pred3<0] = limit_detection[elem]

        R2 = r2_score(Yt,pred3)
        RMSE = np.sqrt(mean_squared_error(Yt,pred3))

        sns.lineplot(x=(np.array(list(range(len(Yt))))*10+5),y=Yt,linestyle='',marker='X',ax=axes[m,i],color=(49/255,130/255,189/255,1),markersize=12)
        sns.lineplot(x=(np.array(list(range(len(Yt))))*10+5), y= pred3,ax=axes[m,i], color=(252/255,141/255,98/255,1))
        mx = max([max(Yt),max(pred3),np.max(pred3)])
        axes[0,i].set_ylabel('')
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        mx = 0.1*mx + mx 
        axes[m,i].set_ylim(0,mx)
        axes[m,i].set_xlabel("")
        axes[m,i].set_xlim(0,np.max((np.array(list(range(len(Yt))))*10+5)+5))
        # axes[m,i].set_title(titles[i+m], fontsize='small', loc='left',fontweight='bold',size=12)
        #axes[0,i].set_title(elem + " ("+units[elem] + ")",fontsize=15)
        axes[m,i].tick_params(axis='x', labelsize=15)
        axes[m,i].tick_params(axis='y', labelsize=15)
        R2text = ('$R^2$ = '+ str(round(R2,2)) + "\n" + "$RMSE$ =" + str(round(RMSE,2)))
        if i == 0:
            if m >= 1:
                axes[m,i].text(x=5,y=mx*0.2,s=R2text,fontsize=15)
            else:
                axes[m,i].text(x=5,y=(mx-mx*0.2),s=R2text,fontsize=15)
        i = i+1
    m = m +1


fig.text(0.01, 0.17, 'Pb (mg/Kg)', va='center', rotation='vertical',fontsize=14)
fig.text(0.01, 0.50, 'Zn (mg/Kg)', va='center', rotation='vertical',fontsize=14)
fig.text(0.01, 0.83, 'Ca (%)', va='center', rotation='vertical',fontsize=14)

fig.text(0.17,0.99, 'BE', va='center', rotation='horizontal',fontsize=16)
fig.text(0.50,0.99, 'P', va='center', rotation='horizontal',fontsize=16)
fig.text(0.83,0.99, 'G', va='center', rotation='horizontal',fontsize=16)

plt.savefig('predicted_elems_woa.png',dpi=300,pad_inches=0)
plt.show()




X = pellet[features]
coreid = ref_for_pellet['coreid']
Y = ref_for_pellet.drop('coreid',axis=1)

#multivariate model for the pellets
#load X data
xdata = tdata[features]

#univariate model for all the dataset
res2 = pd.DataFrame(columns=['R2','RMSE','Predictor'],index=Ycols)
for i in Ycols:
    r = 0
    for k in range(1,8):
        ypred_acum = np.empty(0)
        ytest_acum = np.empty(0)
        xtest_acum = np.empty(0)

        for j in pd.unique(tdata['coreid']):
            xtest = xdata.loc[tdata['coreid']==j,:]
            ytest = (ydata.loc[ydata['coreid']==j,i].to_numpy().astype(float))
            xtrain = xdata.loc[np.logical_not(tdata['coreid']==j),:]
            ytrain = (ydata.loc[np.logical_not(ydata['coreid']==j),i].to_numpy().astype(float))
            mdl = PLSRegression(k)
            mdl.fit(xtrain,ytrain)
            pred = mdl.predict(xtest)
            if (pred<0).any():
                pred[pred<0] = limit_detection[i]
            ypred_acum = np.append(ypred_acum,(pred))
            ytest_acum = np.append(ytest_acum,(ytest))
            xtest_acum = np.append(xtest_acum,xtest)

        r2 = r2_score(ytest_acum,ypred_acum)
        rmse = np.sqrt(mean_squared_error(ytest_acum,ypred_acum))
        if r2 > r:
            r = r2
            rmse2 = rmse
            res = k
    res2.loc[i] = [r,rmse2,res]   
    
res_univariate_all_linear = res2
res_univariate_all_linear.sort_values(by='R2')



#save all the results to csv
pd.concat([res_univariate_all_linear,res_univariate_all_linear_pell],axis=1).sort_index().to_csv("allres.csv")



#Weltje approach for single core and 1cm multivariate
res2 = pd.DataFrame(columns=['Element','R2','RMSE','denom'])

X_train = Xtrain.copy()
Y_train = Ytrain.copy()
Y_test = Ytest.copy()
X_test = Xtest.copy()
for i in units.index:
    if units[i]=='wt%':
        Y_train[i] = Y_train[i]*10000
        Y_test[i] = Y_test[i]*10000

Y_train["R"] = 1000000-Y_train.sum(axis=1)
Y_test["R"] = 1000000-Y_test.sum(axis=1)


X = np.log(X_train.div(gmean(X_train,axis=1),axis=0))
Y = np.log(Y_train.div(gmean(Y_train,axis=1),axis=0))
Xtt = np.log(X_test.div(gmean(X_test,axis=1),axis=0))
Ytt = np.log(Y_test.div(gmean(Y_test,axis=1),axis=0))

x = X.to_numpy()
y = Y.to_numpy()
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(x)

#kernel = DotProduct() + WhiteKernel() + Matern(nu=1.5)
#mdl = GaussianProcessRegressor(kernel=kernel, random_state=0)
mdl = RandomForestRegressor()
mdl.fit(poly_features, y)
Xt = poly.transform(Xtt.to_numpy())
pred3 = mdl.predict((Xt))
pred3 = pd.DataFrame(pred3,columns=Ytt.columns)
pred3 = np.exp(pred3)
pred3 = pred3.multiply(gmean(pred3,axis=1),axis=0)
pred3 = pred3.div(pred3.sum(axis=1),axis=0)*1000000
#pred3 = savgol_filter(pred3.reshape(1,-1),9,3)
#pred3 = pred3[0]
#pred3[pred3<0] = min(pred3[pred3>0])

for j in Y.columns:
    r2 = r2_score(Y_test[j],pred3[j])
    rmse = np.sqrt(mean_squared_error(Y_test[j],pred3[j]))
    res2.loc[len(res2)] = [j,r2,rmse,res]

res_univariate_15BE_linear = res2.sort_values('R2',ascending=False)
res_univariate_15BE_linear.reset_index(inplace=True,drop=True)
res_univariate_15BE_linear.set_index('Element',inplace=True)
