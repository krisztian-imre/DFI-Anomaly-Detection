import csv
import numpy as np
from sklearn.ensemble import IsolationForest

import zipfile
import pandas as pd
import requests
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split 
from matplotlib import pyplot
from numpy.random import normal, gamma
from numpy import hstack
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from sklearn.preprocessing import StandardScaler
from pgmpy.models import BayesianNetwork
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.estimators import BayesianEstimator
import seaborn as sns
import matplotlib.pyplot as plt
from causalinference import CausalModel
import warnings
from dowhy import CausalModel
import pandas as pd
from dowhy import CausalModel
from statsmodels.stats.outliers_influence import variance_inflation_factor
import networkx as nx
warnings.filterwarnings("ignore")

df_ref= pd.read_csv(r'c:\FAOSTAT\reference_europe.csv', encoding='latin1')
# extract only Area Code
df_ref = df_ref[['Area']]
#groupping by Area
df_ref = df_ref.groupby('Area').size().reset_index(name='counts')

#pesticides data
zip_file_path = r'c:\FAOSTAT\Inputs_Pesticides_Use_E_All_Data_(Normalized).zip'
file_path_inside_zip = 'Inputs_Pesticides_Use_E_All_Data_(Normalized).csv'  # Provide the correct path inside the ZIP archive
# Extract the CSV file from the ZIP archive
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extract(file_path_inside_zip, r'c:\FAOSTAT_extract')  # Provide the correct path to extract the file
# Load the CSV file into a DataFrame
df = pd.read_csv(r'c:\FAOSTAT_extract\Inputs_Pesticides_Use_E_All_Data_(Normalized).csv', encoding='latin1')
#filtered pesticides data
filtered_pest = df.loc[(df['Element'] == 'Agricultural Use')]
filtered_pest = filtered_pest.loc[(df['Item'] == 'Pesticides (total)')]
# row filter with reference list
filtered_pest = filtered_pest[filtered_pest['Area'].isin(df_ref['Area'])]


# fertilizers data
zip_file_path = r'c:\FAOSTAT\Inputs_FertilizersNutrient_E_All_Data_(Normalized).zip'
file_path_inside_zip = 'Inputs_FertilizersNutrient_E_All_Data_(Normalized).csv'  # Provide the correct path inside the ZIP archive
# Extract the CSV file from the ZIP archive
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extract(file_path_inside_zip, r'c:\FAOSTAT_extract')  # Provide the correct path to extract the file
# Load the CSV file into a DataFrame
df = pd.read_csv(r'c:\FAOSTAT_extract\Inputs_FertilizersNutrient_E_All_Data_(Normalized).csv', encoding='latin1')

#filtered fertilizers data_Nitrogen
filtered_fert_N = df.loc[(df['Element'] == 'Agricultural Use')]
filtered_fert_N = filtered_fert_N.loc[(df['Item'] == 'Nutrient nitrogen N (total)')]

# row filter with reference list
filtered_fert_N = filtered_fert_N[filtered_fert_N['Area'].isin(df_ref['Area'])]


# Sum the values by Area and Year
#filtered_fert_N = filtered_fert_N.groupby(['Area', 'Year']).sum().reset_index()

# filtered fertilizers data_Phosphorus
filtered_fert_P = df.loc[(df['Element'] == 'Agricultural Use')]
filtered_fert_P = filtered_fert_P.loc[(df['Item'] == 'Nutrient phosphate P2O5 (total)')]

# row filter with reference list
filtered_fert_P = filtered_fert_P[filtered_fert_P['Area'].isin(df_ref['Area'])]

# Sum the values by Area and Year
#filtered_fert_P = filtered_fert_P.groupby(['Area', 'Year']).sum().reset_index()


# filtered fertilizers data_Potassium
filtered_fert_K = df.loc[(df['Element'] == 'Agricultural Use')]
filtered_fert_K = filtered_fert_K.loc[(df['Item'] == 'Nutrient potash K2O (total)')]
# row filter with reference list
filtered_fert_K = filtered_fert_K[filtered_fert_K['Area'].isin(df_ref['Area'])]

# Sum the values by Area and Year
#filtered_fert_K = filtered_fert_K.groupby(['Area', 'Year']).sum().reset_index()

#joiner
joiner = pd.merge(filtered_fert_P, filtered_fert_N, on=['Area','Year'], how='inner')
joinerNPK = pd.merge(joiner, filtered_fert_K, on=['Area','Year'], how='inner')




#Assuming 'joiner' is your DataFrame
columns_to_delete = ['Area Code_x', 'Area Code (M49)','Area Code (M49)_x', 'Item Code_x', 'Item_x',
       'Element Code_x', 'Element_x', 'Year Code_x', 'Unit_x',
        'Flag_x', 'Area Code_y', 'Area Code (M49)_y', 'Item Code','Item','Item Code_y',
       'Item_y', 'Element Code_y', 'Element_y', 'Year Code_y', 'Unit_y',
        'Flag_y', 'Note_x','Note_y','Area Code','Element','Element Code','Year Code','Unit','Flag']

joinerNPK.drop(columns=columns_to_delete, inplace=True, axis=1)
#column filter
joinerNPK.rename(columns={'Value_x': 'Phosphorus', 'Value_y': 'Nitrogen', 'Value': 'Potassium'}, inplace=True)
# mean of NPK
joinerNPK['fertilizer'] = joinerNPK[['Phosphorus', 'Nitrogen', 'Potassium']].sum(axis=1)
 #delete columns
columns_to_delete = ['Phosphorus', 'Nitrogen', 'Potassium','Note']
joinerNPK.drop(columns=columns_to_delete, inplace=True, axis=1)
  

#joiner
joiner1 = pd.merge(joinerNPK, filtered_pest, on=['Area','Year'], how='inner')   
#Assuming 'joiner' is your DataFrame
columns_to_delete = ['Area Code', 'Item Code', 'Element Code', 'Area Code (M49)', 'Item','Element','Year Code', 'Unit', 'Flag']
joiner1.drop(columns=columns_to_delete, inplace=True) 
#column filter
joiner1.rename(columns={'Value': 'Pesticides'}, inplace=True)
    

# land use data
land_use= pd.read_csv('land-use-over-the-long-term.csv', encoding='latin1')

#rename Entity to Area
land_use.rename(columns={'Entity': 'Area'}, inplace=True)

# Sum the Land use: 'Built-up area', 'Land use: Grazingland', 'Land use: Cropland' a new column 'land_use'
land_use['land_use'] = land_use[['Land use: Built-up area', 'Land use: Grazingland', 'Land use: Cropland']].sum(axis=1)

# Assuming 'joiner' is your DataFrame
columns_to_delete = ['Land use: Built-up area', 'Land use: Grazingland', 'Land use: Cropland']
land_use.drop(columns=columns_to_delete, inplace=True)

# drop 'code' column
land_use.drop(columns='Code', inplace=True)

# Assuming 'joiner' is your DataFrame
columns_to_delete = ['Code']



# joiner
joiner3 = pd.merge(land_use, joiner1, on=['Area','Year'], how='inner') 




# Maize
zip_file_path = r'c:\FAOSTAT\Production_Crops_Livestock_E_All_Data_(Normalized).zip'
file_path_inside_zip = 'Production_Crops_Livestock_E_All_Data_(Normalized).csv'  # Provide the correct path inside the ZIP archive
# Extract the CSV file from the ZIP archive
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extract(file_path_inside_zip, r'c:\FAOSTAT_extract')  # Provide the correct path to extract the file
# Load the CSV file into a DataFrame
df_o = pd.read_csv(r'c:\FAOSTAT_extract\Production_Crops_Livestock_E_All_Data_(Normalized).csv', encoding='latin1')
#filtered maize data
filtered_df_m = df_o.loc[(df_o['Item'] == 'Maize (corn)')]
filtered_df_m = filtered_df_m.loc[(df_o['Element'] == 'Yield')]

# joiner
joiner4 = pd.merge(filtered_df_m, joiner3, on=['Area','Year'], how='inner')

# Assuming 'joiner' is your DataFrame
columns_to_delete = ['Area Code','Area Code (M49)','Item Code','Item Code (CPC)','Item','Element Code', 'Element','Year Code', 'Unit', 'Flag', 'Note_x', 'Note_y']

joiner4.drop(columns=columns_to_delete, inplace=True)
#column filter
joiner4.rename(columns={'Value': 'Maize'}, inplace=True)



#Temperature data
zip_file_path = r'c:\FAOSTAT\Environment_Temperature_change_E_All_Data_(Normalized).zip'
file_path_inside_zip = 'Environment_Temperature_change_E_All_Data_(Normalized).csv'  # Provide the correct path inside the ZIP archive
# Extract the CSV file from the ZIP archive
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extract(file_path_inside_zip, r'c:\FAOSTAT_extract')  # Provide the correct path to extract the file
# Load the CSV file into a DataFrame
df = pd.read_csv(r'c:\FAOSTAT_extract\Environment_Temperature_change_E_All_Data_(Normalized).csv', encoding='latin1')
#filtered temperature data
filtered_temp = df.loc[(df['Months'] == 'Meteorological year')]
filtered_temp = filtered_temp.loc[(df['Element'] == 'Temperature change')]
# row filter with reference list
filtered_temp = filtered_temp[filtered_temp['Area'].isin(df_ref['Area'])]
columns_to_delete = ['Area Code', 'Area Code (M49)','Months', 'Months Code', 'Element Code', 'Element','Year Code', 'Unit', 'Flag']
filtered_temp.drop(columns=columns_to_delete, inplace=True)
#column filter
filtered_temp.rename(columns={'Value': 'Temperature change'}, inplace=True)

#joiner
joiner5 = pd.merge(joiner4, filtered_temp, on=['Area','Year'], how='inner')


#population data
zip_file_path = r'c:\FAOSTAT\Population_E_All_Data_(Normalized).zip'
file_path_inside_zip = 'Population_E_All_Data_(Normalized).csv'  # Provide the correct path inside the ZIP archive
# Extract the CSV file from the ZIP archive
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extract(file_path_inside_zip, r'c:\FAOSTAT_extract')  # Provide the correct path to extract the file
# Load the CSV file into a DataFrame
df = pd.read_csv(r'c:\FAOSTAT_extract\Population_E_All_Data_(Normalized).csv', encoding='latin1')
#filtered population data
filtered_pop = df.loc[(df['Element'] == 'Total Population - Both sexes')]
# row filter with reference list
filtered_pop = filtered_pop[filtered_pop['Area'].isin(df_ref['Area'])]   
 
#Assuming 'joiner' is your DataFrame
columns_to_delete = ['Area Code','Area Code (M49)','Item Code','Item','Element','Element Code', 'Year Code', 'Unit', 'Flag', 'Note']
filtered_pop.drop(columns=columns_to_delete, inplace=True)
#column filter
filtered_pop.rename(columns={'Value': 'Population'}, inplace=True)
#joiner
joiner6 = pd.merge(joiner5, filtered_pop, on=['Area','Year'], how='inner')


# Rape or colza seed
zip_file_path = r'c:\FAOSTAT\Production_Crops_Livestock_E_All_Data_(Normalized).zip'
file_path_inside_zip = 'Production_Crops_Livestock_E_All_Data_(Normalized).csv'  # Provide the correct path inside the ZIP archive
# Extract the CSV file from the ZIP archive
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extract(file_path_inside_zip, r'c:\FAOSTAT_extract')  # Provide the correct path to extract the file
# Load the CSV file into a DataFrame
df_o = pd.read_csv(r'c:\FAOSTAT_extract\Production_Crops_Livestock_E_All_Data_(Normalized).csv', encoding='latin1')
#filtered rapeseed data
filtered_df_r = df_o.loc[(df_o['Item'] == 'Rape or colza seed')]
filtered_df_r = filtered_df_r.loc[(df_o['Element'] == 'Yield')]

# joiner
joiner7 = pd.merge(filtered_df_r, joiner6, on=['Area','Year'], how='inner') 

# Assuming 'joiner' is your DataFrame
columns_to_delete = ['Area Code','Area Code (M49)','Item Code','Item Code (CPC)','Item','Element Code', 'Element','Year Code', 'Unit', 'Flag','Note']
joiner7.drop(columns=columns_to_delete, inplace=True)
#column filter
joiner7.rename(columns={'Value': 'Rapeseed'}, inplace=True)


#load precipitation data
df = pd.read_csv('precip.csv', encoding='latin1')

#joiner
joiner8 = pd.merge(joiner7, df, on=['Area','Year'], how='left')


#load renewable_freshwater data
df = pd.read_csv('renewable_freshwater.csv', encoding='latin1')

#joiner
joiner9 = pd.merge(joiner8, df, on=['Area','Year'], how='left')


#load ghg_emisions
df = pd.read_csv('ghg_emissions.csv', encoding='latin1')

#joiner
joiner10= pd.merge(joiner9, df, on=['Area','Year'], how='left')


#Assuming 'joiner' is your DataFrame
columns_to_delete = ['Area', 'Year']
joiner10.drop(columns=columns_to_delete, inplace=True)
#column filter
joiner10.rename(columns={'Value': 'GHG emissions'}, inplace=True)

#filter columns
joiner10_filter_for_prec = joiner10[['Population', 'Pesticides', 'fertilizer','precip']]
#print(joiner10)
joiner10_filter_for_rfw = joiner10[['Population', 'Pesticides', 'fertilizer','rfw']]

joiner10_filter_for_ghg = joiner10[['fertilizer','ghg_emissions']]

#joiner9_filter_for_prec_rfw_ghg= joiner9[['Population', 'Pesticides', 'fertilizer','precip','rfw','ghg_emissions']]

#remove missing values
joiner10_miss = joiner10.dropna()


import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# heatmap among variables
import seaborn as sns
import matplotlib.pyplot as plt

#impute prec missing values
imp = IterativeImputer(max_iter=10, random_state=0)
imp.fit(joiner10_filter_for_prec)
imputed_data = imp.transform(joiner10_filter_for_prec)
joiner10_imp_prec = pd.DataFrame(imputed_data, columns=joiner10_filter_for_prec.columns)

#impute rfw missing values
imp = IterativeImputer(max_iter=10, random_state=0)
imp.fit(joiner10_filter_for_rfw)
imputed_data = imp.transform(joiner10_filter_for_rfw)
joiner10_imp_rfw= pd.DataFrame(imputed_data, columns=joiner10_filter_for_rfw.columns)

#impute ghg missing values
imp = IterativeImputer(max_iter=10, random_state=0)
imp.fit(joiner10_filter_for_ghg)
imputed_data = imp.transform(joiner10_filter_for_ghg)
joiner10_imp_ghg = pd.DataFrame(imputed_data, columns=joiner10_filter_for_ghg.columns)


#Column filter

joiner10_imp_prec = joiner10_imp_prec[['precip']]
joiner10_imp_rfw = joiner10_imp_rfw[['rfw']]
joiner10_imp_ghg = joiner10_imp_ghg[['ghg_emissions']]

#joiner6 with joiner10_imp_prec
joiner7 = pd.merge(joiner7, joiner10_imp_prec, left_index=True, right_index=True)

#joiner6 with joiner10_imp_rfw
joiner7 = pd.merge(joiner7, joiner10_imp_rfw, left_index=True, right_index=True)

#joiner6 with joiner10_imp_ghg
joiner7 = pd.merge(joiner7, joiner10_imp_ghg, left_index=True, right_index=True)

# to csv
joiner7.to_csv('year_area_joiner7.csv', index=False)

#Drop Area and Year columns
joiner7.drop(columns=['Area','Year'], inplace=True)

#Drop Population column
#joiner7.drop(columns=['Population'], inplace=True)

#column names change
joiner7.rename(columns={'Pesticides': 'Pesticides', 'fertilizer': 'Fertilizer', 'precip': 'Precipitation', 'rfw': 'Renewable Freshwater', 'ghg_emissions': 'GHG Emissions', 'Population': 'Population','Maize': 'Maize', 'Rapeseed': 'Rapeseed','Temperature_change': 'Temperature Change', 'land_use': 'Land Use'}, inplace=True)

# Write to Excel file
joiner7.to_excel('input_dataset.xlsx', index=False)
joiner7.to_csv('input_dataset.csv', index=False)
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Set contamination level without grid search
contamination_level = 0.03

# Fill NaN values with the mean of each column
joiner7_clean = joiner7.fillna(joiner7.mean())

# Initialize and fit Isolation Forest model
isolation_forest = IsolationForest(contamination=contamination_level, random_state=42)
isolation_forest.fit(joiner7_clean)

# Obtain anomaly scores for the entire dataset
anomaly_scores = -isolation_forest.decision_function(joiner7_clean)  # Inverse the anomaly scores

# Determine threshold based on the contamination level
threshold = np.percentile(anomaly_scores, 100 * (1 - contamination_level))

# Determine anomalies based on the threshold
predictions = (anomaly_scores > threshold).astype(int)

# Calculate Silhouette Score
if len(set(predictions)) > 1:  # Only evaluate if there are multiple classes
    best_score = silhouette_score(joiner7_clean, predictions)
else:
    best_score = -1  # Invalid score if only one class found

# Set best parameters
best_params = {'contamination': contamination_level, 'threshold': threshold}

# Print the best parameters and the best Silhouette score
print(f"Best Contamination Level: {best_params['contamination']}")
print(f"Best Threshold: {best_params['threshold']}")
print(f"Best Silhouette Score: {best_score}")

# Fit the model with the best parameters on the entire dataset
best_isolation_forest = IsolationForest(contamination=best_params['contamination'], random_state=42)
best_isolation_forest.fit(joiner7_clean)

# Obtain anomaly scores using the best model
best_anomaly_scores = -best_isolation_forest.decision_function(joiner7_clean)  # Inverse the anomaly scores

# Determine anomalies based on the threshold
best_predictions = (best_anomaly_scores > best_params['threshold']).astype(int)
# Visualize the anomaly scores with a modern color palette
plt.figure(figsize=(10, 6), dpi=600)
sns.histplot(best_anomaly_scores, bins=50, kde=False, color="#9BB7BB", label='Histogram')  # Modern teal blue
# Plot KDE separately with a salmon color
sns.kdeplot(best_anomaly_scores, color="#0F0B0A", linewidth=2, label='KDE')  # Salmon color
plt.axvline(best_params['threshold'], color='#FA8072', linestyle='--', label='Threshold')  # Salmon color
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Visualize the anomalies with custom colors for anomaly and normal points
plt.figure(figsize=(10, 6), dpi=600)
colors = np.where(best_predictions == 1, "#FA8072", "#9BB7BB")  # anomalies: salmon, normal: teal blue
plt.scatter(range(len(best_anomaly_scores)), best_anomaly_scores, c=colors, edgecolor='k')
plt.axhline(best_params['threshold'], color='#0F0B0A', linestyle='--', label='Threshold')  # dark line for threshold
plt.xlabel('Data Point Index')
plt.ylabel('Anomaly Score')
plt.legend()
plt.show()


# Perform PCA to find the strongest common variance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize the data
scaler = StandardScaler()
joiner7_scaled = scaler.fit_transform(joiner7_clean)

# Fit PCA
pca = PCA()
pca.fit(joiner7_scaled)

# Plot explained variance ratio
plt.figure(figsize=(10, 6), dpi=600)
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o', linestyle='--')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio by Principal Component')
plt.show()


# Select the number of components that explain most of the variance
n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
print(f"Number of components explaining 95% of the variance: {n_components}")

# Transform the data using the selected number of components
pca = PCA(n_components=n_components)
joiner7_pca = pca.fit_transform(joiner7_scaled)

print(len(joiner7_pca))
print(len(joiner7_pca[0]))

# Perform t-SNE for visualization
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.offline as pyo

# Fit t-SNE on the PCA-transformed data
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(joiner7_pca)

# Perform 3D t-SNE for visualization
tsne_3d = TSNE(n_components=3, random_state=42)
X_tsne_3d = tsne_3d.fit_transform(joiner7_pca)

# Combine t-SNE results with anomaly scores and original data
tsne_df = pd.DataFrame(X_tsne_3d, columns=['t-SNE Dimension 1', 't-SNE Dimension 2', 't-SNE Dimension 3'])
tsne_df['Anomaly Score'] = best_anomaly_scores
tsne_df = pd.concat([tsne_df, joiner7.reset_index(drop=True)], axis=1)

print(tsne_df)

# Create interactive 3D t-SNE plot and save as HTML (red = higher anomaly score)
# Note: Higher anomaly scores appear as white/light colors, lower scores as red/dark colors
fig = go.Figure(data=[go.Scatter3d(
    x=X_tsne_3d[:, 0],
    y=X_tsne_3d[:, 1],
    z=X_tsne_3d[:, 2],
    mode='markers',
    marker=dict(
        size=5,
        color=best_anomaly_scores,
        colorscale='RdBu_r',  # Red for high anomaly scores, blue for low
        colorbar=dict(title="Anomaly Score"),
        line=dict(width=0.5, color='DarkSlateGrey')
    ),
    text=[f'Point {i}<br>Anomaly Score: {score:.3f}' for i, score in enumerate(best_anomaly_scores)],
    hovertemplate='<b>%{text}</b><br>t-SNE 1: %{x:.2f}<br>t-SNE 2: %{y:.2f}<br>t-SNE 3: %{z:.2f}<extra></extra>'
)])

fig.update_layout(
    title='3D t-SNE Visualization with Anomaly Scores',
    scene=dict(
        xaxis_title='t-SNE Dimension 1',
        yaxis_title='t-SNE Dimension 2',
        zaxis_title='t-SNE Dimension 3'
    ),
    width=900,
    height=700,
    margin=dict(r=20, b=10, l=10, t=40)
)

# Save as standalone HTML file that can be opened on any computer
pyo.plot(fig, filename='tsne_3d_anomaly_visualization.html', auto_open=False, include_plotlyjs=True)
print("3D t-SNE plot saved as 'tsne_3d_anomaly_visualization.html' (standalone file)")

# column filter t-sne and anomaly score columns, sorted by decreasing anomaly score
tsne_df = tsne_df[['t-SNE Dimension 1', 't-SNE Dimension 2', 't-SNE Dimension 3', 'Anomaly Score']].sort_values(by='Anomaly Score', ascending=False)


# Visualize the 3D t-SNE results as a 2D matrix of pairwise projections
fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=600)

# 2D projections: (1,2), (1,3), (2,3)
axes[0].scatter(tsne_df['t-SNE Dimension 1'], tsne_df['t-SNE Dimension 2'], c=tsne_df['Anomaly Score'], cmap='coolwarm', edgecolor='k')
axes[0].set_xlabel('t-SNE Dimension 1')
axes[0].set_ylabel('t-SNE Dimension 2')
axes[0].set_title('t-SNE 1 vs 2')

axes[1].scatter(tsne_df['t-SNE Dimension 1'], tsne_df['t-SNE Dimension 3'], c=tsne_df['Anomaly Score'], cmap='coolwarm', edgecolor='k')
axes[1].set_xlabel('t-SNE Dimension 1')
axes[1].set_ylabel('t-SNE Dimension 3')
axes[1].set_title('t-SNE 1 vs 3')

axes[2].scatter(tsne_df['t-SNE Dimension 2'], tsne_df['t-SNE Dimension 3'], c=tsne_df['Anomaly Score'], cmap='coolwarm', edgecolor='k')
axes[2].set_xlabel('t-SNE Dimension 2')
axes[2].set_ylabel('t-SNE Dimension 3')
axes[2].set_title('t-SNE 2 vs 3')

plt.tight_layout()
plt.colorbar(axes[2].collections[0], ax=axes, label='Anomaly Score', orientation='vertical', fraction=0.02, pad=0.04)
plt.show()



# Filter anomaly scores greater than 0
tsne_df7 = tsne_df[(tsne_df['Anomaly Score'] > 0)]
print(tsne_df7)
print(len(tsne_df7))

# Visualize the anomalies in the 3D t-SNE space with a color transition based on anomaly scores
fig = plt.figure(figsize=(10, 8), dpi=600)
ax = fig.add_subplot(111, projection='3d')

# Mark points with anomaly scores > 0 using a different symbol (e.g., rectangles)
sc = ax.scatter(
    X_tsne_3d[:, 0], 
    X_tsne_3d[:, 1], 
    X_tsne_3d[:, 2], 
    c=best_anomaly_scores, 
    cmap='coolwarm', 
    edgecolor='k'
)

# Highlight points with anomaly scores > 0 using rectangles
anomalies = tsne_df7.index
ax.scatter(
    X_tsne_3d[anomalies, 0], 
    X_tsne_3d[anomalies, 1], 
    X_tsne_3d[anomalies, 2], 
    c='red', 
    marker='s', 
    label='Anomalies (Score > 0)'
)

fig.colorbar(sc, ax=ax, label='Anomaly Score')
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.set_zlabel('t-SNE Dimension 3')
ax.set_title('Anomalies Detected by Isolation Forest (3D t-SNE)')
ax.legend()
plt.show()



# Mean of the anomaly scores
print(f"Mean Inverse Anomaly Score: {np.mean(best_anomaly_scores)}")

# Number of anomalies detected
print(f"Number of Anomalies Detected: {np.sum(best_predictions)}")

# Percentage of anomalies detected
print(f"Percentage of Anomalies Detected: {np.mean(best_predictions) * 100:.2f}%")

import shap
import matplotlib
import numpy as np
import pandas as pd

# Initialize the SHAP explainer
explainer = shap.TreeExplainer(best_isolation_forest)

# Calculate SHAP values
shap_values = explainer.shap_values(joiner7_clean)

# Visualize SHAP values for the first few instances with 600 dpi
plt.figure(dpi=600)
shap.summary_plot(shap_values, joiner7_clean, show=False)
plt.show()

# Visualize SHAP values bar plot with individual feature colors, sorted by highest SHAP values at the top
feature_names = joiner7.columns
# Assign a unique color to each feature
colors = matplotlib.cm.get_cmap('tab20', len(feature_names)).colors
color_map = dict(zip(feature_names, colors))

# Compute mean absolute SHAP values and sort descending (top = highest SHAP values)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
sorted_idx = np.argsort(mean_abs_shap)[::-1]
sorted_features = feature_names[sorted_idx]
sorted_shap = mean_abs_shap[sorted_idx]
sorted_colors = [color_map[f] for f in sorted_features]

# Calculate SHAP value distribution by feature
shap_dist = {}
for i, feat in enumerate(feature_names):
    vals = shap_values[:, i]
    strongly_neg = np.sum(vals <= -1.0)
    moderately_neg = np.sum((vals > -1.0) & (vals <= -0.5))
    mildly_neg = np.sum((vals > -0.5) & (vals <= -0.2))
    neutral = np.sum((vals > -0.2) & (vals < 0.2))
    shap_dist[feat] = {
        'strongly_negative': strongly_neg,
        'moderately_negative': moderately_neg,
        'mildly_negative': mildly_neg,
        'neutral': neutral,
        'total': len(vals)
    }

# Prepare SHAP value classification table (rows: classification, columns: features, values: percent rate)

# Define the classification labels
classification_labels = ['strongly_negative', 'moderately_negative', 'mildly_negative', 'neutral']

# Prepare a DataFrame to hold the percent rates
shap_percent_table = pd.DataFrame(index=classification_labels, columns=feature_names)

# Fill the table with percent rates
for feat in feature_names:
    total = shap_dist[feat]['total']
    for label in classification_labels:
        shap_percent_table.loc[label, feat] = 100 * shap_dist[feat][label] / total if total > 0 else 0

# Rename the index for better display
shap_percent_table.index = [
    'Strongly negative (<= -1.0)',
    'Moderately negative (-1.0 to -0.5)',
    'Mildly negative (-0.5 to -0.2)',
    'Neutral (-0.2 to 0.2)'
]

print("SHAP Value Classification Table (percent rates):")
print(shap_percent_table.round(2).astype(str) + '%')



# Example: Visualize distribution for one feature with red (negative) and blue (positive) points
import matplotlib.pyplot as plt

for i, feat in enumerate(feature_names):
    vals = shap_values[:, i]
    plt.figure(figsize=(8, 2), dpi=600)
    colors = ['red' if v < 0 else 'blue' for v in vals]
    plt.scatter(range(len(vals)), vals, c=colors, alpha=0.6)
    plt.axhline(-1.0, color='gray', linestyle='--', linewidth=0.8)
    plt.axhline(-0.5, color='gray', linestyle='--', linewidth=0.8)
    plt.axhline(-0.2, color='gray', linestyle='--', linewidth=0.8)
    plt.axhline(0.2, color='gray', linestyle='--', linewidth=0.8)
    plt.title(f"SHAP Value Distribution for {feat}")
    plt.xlabel("Sample Index")
    plt.ylabel("SHAP Value")
    plt.tight_layout()
    plt.show()

# Print the SHAP values
print("SHAP values:\n", shap_values)

# Use salmon color for all bars
salmon_color = "#FA8072"
plt.figure(figsize=(10, 6), dpi=600)
plt.barh(range(len(sorted_features)), sorted_shap, color=salmon_color)
plt.yticks(range(len(sorted_features)), sorted_features)
plt.xlabel('Mean(|SHAP value|)')
plt.gca().invert_yaxis()  # Highest SHAP values at the top

# Annotate each bar with its value
for i, v in enumerate(sorted_shap):
    plt.text(v + max(sorted_shap)*0.01, i, f"{v:.3f}", va='center', color='black', fontsize=10)

plt.tight_layout()
plt.show()




# Visualize force plot for the first instance in tsne_df1
# shap.initjs()
# Filter for t-SNE Dimension 1 > 5 and t-SNE Dimension 2 < 4
tsne_df1 = tsne_df[(tsne_df['t-SNE Dimension 1'] > 5) & (tsne_df['t-SNE Dimension 2'] < 4)]

# If there are any such points, visualize force plot for the first instance
if not tsne_df1.empty:
    idx = tsne_df1.index[0]
    force_plot_1 = shap.force_plot(explainer.expected_value, shap_values[idx, :], joiner7.iloc[idx, :])
    shap.save_html("force_plot_1.html", force_plot_1)

# Filter for t-SNE Dimension 1 > 5 and t-SNE Dimension 3 < 4
tsne_df2 = tsne_df[(tsne_df['t-SNE Dimension 1'] > 5) & (tsne_df['t-SNE Dimension 3'] < 4)]

# If there are any such points, visualize force plot for the first instance
if not tsne_df2.empty:
    idx2 = tsne_df2.index[0]
    force_plot_2 = shap.force_plot(explainer.expected_value, shap_values[idx2, :], joiner7.iloc[idx2, :])
    shap.save_html("force_plot_2.html", force_plot_2)
# 

# Visualize force plot for the first instance in tsne_df7
force_plot_7 = shap.force_plot(explainer.expected_value, shap_values[tsne_df7.index[0], :], joiner7.iloc[tsne_df7.index[0], :])
shap.save_html("force_plot_7.html", force_plot_7)



# Visualize force plot for the first instance in the dataset
force_plot_all = shap.force_plot(explainer.expected_value, shap_values, joiner7)
shap.save_html("force_plot_all.html", force_plot_all)

# Waterfall plot for the first instance in tsne_df7, aligning direction so blue shows SHAP values decreasing (negative impact)
# Use original SHAP values and base value so blue bars indicate features decreasing the anomaly score
shap_exp_7 = shap.Explanation(
    values=shap_values[tsne_df7.index[0], :],
    base_values=explainer.expected_value,
    data=joiner7.iloc[tsne_df7.index[0], :],
    feature_names=joiner7.columns
)
plt.figure(dpi=600)
shap.waterfall_plot(shap_exp_7)
plt.show()


