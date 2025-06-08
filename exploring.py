import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import PercentFormatter

# Equivalent to `summarytools::freq` for frequency tables
def freq_table(series, report_nas=False, order='freq'):
    if report_nas:
        counts = series.value_counts(dropna=False)
    else:
        counts = series.value_counts(dropna=True)

    percentages = counts / len(series) * 100
    df = pd.DataFrame({'Count': counts, 'Percentage': percentages})

    if order == 'freq':
        df = df.sort_values(by='Count', ascending=False)
    elif order == 'asc':
        df = df.sort_values(by='Count', ascending=True)
    elif order == 'desc':
        df = df.sort_values(by='Count', ascending=False)
    return df

# Data Set EDIT 2021 ------------------------------------------------------

# Set working directory (adjust as per your system)
os.chdir('C:\\Users\\USUARIO\\Desktop\\Trabajo\\Iberoamericana\\Investigación\\EDIT')
print(os.getcwd())
print(os.listdir())

EDIT_19_20 = pd.read_csv("Datos\\EDIT_X_2019_2020.csv", sep=';')
print(EDIT_19_20.shape) # 6798 empresas y 729 variables

print(EDIT_19_20.columns)
# Número de orden (identificador)
print(EDIT_19_20['NORDEMP'].duplicated().sum())
print(EDIT_19_20['NORDEMP'].duplicated().any())

# Numero de empresas investigadas según actividad económica

print(EDIT_19_20['CIIU4'].nunique()) # 55 actividades económicas

print(EDIT_19_20['CIIU4'].sort_values().unique()) # Códigos de las actividades

# Actividad económica
print(EDIT_19_20['CIIU4'].value_counts())
tabla_freq = freq_table(EDIT_19_20['CIIU4'], report_nas=False, order='freq')
print(tabla_freq)

# Prueba barras apiladas------------

tabla = EDIT_19_20.groupby('CIIU4').size().reset_index(name='Count')
tabla['Percentage'] = (tabla['Count'] / tabla['Count'].sum()) * 100
tabla['Percentage'] = tabla['Percentage'].round(2)

# Mapping CIIU4 codes to names
ciiu4_mapping = {
    108: 'Alimentos',
    141: 'Confección',
    152: 'Calzado',
    181: 'Impresión',
    222: 'Plásticos',
    239: 'Minerales (no-metal)',
    251: 'Metales (estructuras)',
    259: 'Metales (otros)',
    311: 'Muebles'
}
tabla['CIIU4'] = tabla['CIIU4'].replace(ciiu4_mapping)

# Gráfico presentación (%)
# Sort by Percentage and take top 9
tabla_plot = tabla.sort_values(by='Percentage', ascending=False).head(9)

plt.figure(figsize=(10, 7))
sns.barplot(y='CIIU4', x='Percentage', data=tabla_plot, palette='Oranges_r')
plt.xlabel("Porcentaje de empresas")
plt.ylabel("Categoría CIUU4")
plt.title('Sectores (CIIU4) con más empresas registradas (50.68%)')
plt.figtext(0.99, 0.01, 'Fuente: Elaboración propia. Datos EDIT (2019-2020)', ha='right', fontsize=9)

for index, row in tabla_plot.iterrows():
    plt.text(row['Percentage'] + 0.5, index, f"{row['Percentage']}", va='center') # Adjust x-offset for label

plt.tight_layout()
plt.show()

# TIPOLOGÍA -----------------
# DANE las clasifica en 4 tipos en función de los resultados
# Innovadoras en sentido estricto (11)
# Innovadoras en sentido amplio (1561)
# Potencialmente innovadoras (278)
# No innovadoras (5470)

print((EDIT_19_20['TIPOLO'].value_counts(normalize=True) * 100).round(1))

print(EDIT_19_20['TIPOLO'].astype('category').cat.categories)

# Aparece la categoría "INTENC" que, siguiendo el boletín
# debe corresponder a "NOINNO"
EDIT_19_20['TIPOLO'] = EDIT_19_20['TIPOLO'].replace("INTENC", "NOINNO")

# Gráfico de barras: tipología
df_tipologia = EDIT_19_20.groupby('TIPOLO').size().reset_index(name='frecuencia')
df_tipologia['relativa'] = 100 * (df_tipologia['frecuencia'] / len(EDIT_19_20))
df_tipologia = df_tipologia.sort_values(by='relativa', ascending=True)

# Map labels for x-axis
tipologia_labels = {
    "ESTRICTO": "Estrictos",
    "POTENC": "Potenciales",
    "AMPLIA": "Amplios",
    "NOINNO": "No innovadores"
}
df_tipologia['TIPOLO_label'] = df_tipologia['TIPOLO'].map(tipologia_labels)

plt.figure(figsize=(10, 7))
ax = sns.barplot(x='TIPOLO_label', y='relativa', data=df_tipologia, hue='TIPOLO_label', palette='Spectral', dodge=False, legend=False)
plt.xlabel("Tipología de innovación")
plt.ylabel("Frecuencia relativa")
plt.title("Tipología de innovación según el DANE (2019-2020)")
plt.figtext(0.99, 0.01, 'Fuente: Elaboración propia. Datos EDIT X 2021', ha='right', fontsize=9)
plt.gca().yaxis.set_major_formatter(PercentFormatter())

for container in ax.containers:
    ax.bar_label(container, fmt='%.1f%%', label_type='edge', padding=3)

plt.tight_layout()
plt.show()

# CIIU y tipologia --------------------------------------------------------

print(EDIT_19_20['TIPOLO'].value_counts())
print(EDIT_19_20['CIIU4'].value_counts())
print(EDIT_19_20['CIIU4'].unique())

df_ciiu_tipologia = pd.DataFrame({
    'CIIU4': EDIT_19_20['CIIU4'],
    'Tipologia': EDIT_19_20['TIPOLO']
})

# Recalculate prop.table as in R with margin=2 (column-wise percentages)
contin = pd.crosstab(df_ciiu_tipologia['Tipologia'], df_ciiu_tipologia['CIIU4'], normalize='columns')
contin = contin.stack().reset_index(name='Freq')
contin.columns = ['Tipologia', 'CIIU4', 'Freq']

print(contin.sample(5)) # Vista aleatoria de mi df

# Sectores mas innovadores en sentido amplio
df_amplia = contin[contin['Tipologia'] == 'AMPLIA'].sort_values(by='Freq', ascending=False).head(10)

unique_ciiu4_amplia = df_amplia['CIIU4'].unique()
contin_filtered = contin[contin['CIIU4'].isin(unique_ciiu4_amplia)].copy()

# Map CIIU4 codes to descriptive names
nom_levs_mapping = {
    101: 'Carnes',
    103: 'Frutas, legumbres(otros)',
    104: 'Aceites',
    105: 'Lácteos',
    106: 'Alimentos animales',
    202: 'Plaguicidas',
    206: 'Jabones y detergentes',
    210: 'Farmacéuticos',
    262: 'Informáticos',
    291: 'Vehículos'
}

contin_filtered['CIIU4'] = contin_filtered['CIIU4'].replace(nom_levs_mapping)
# Ensure the order for plotting
order_ciiu4 = df_amplia['CIIU4'].replace(nom_levs_mapping).tolist()

plt.figure(figsize=(12, 8))
ax = sns.barplot(x='Freq', y='CIIU4', hue='Tipologia', data=contin_filtered, palette='Set1', order=order_ciiu4)
plt.xlabel("Porcentaje")
plt.ylabel("CIIU4")
plt.title("Sectores CIIU4 con más innovación en sentido amplio")
plt.figtext(0.99, 0.01, 'Fuente: Elaboración propia. Datos EDIT X 2021', ha='right', fontsize=9)
plt.gca().xaxis.set_major_formatter(PercentFormatter(1)) # Scale in percentage
plt.legend(title='Tipología')

# Add text labels
for container in ax.containers:
    ax.bar_label(container, fmt='%.0f%%', label_type='center', color='white', fontsize=10)

plt.tight_layout()
plt.show()

# Sectores mas innovadores en sentido estricto

print(EDIT_19_20['TIPOLO'].value_counts()) # Solamente hay 11 empresas innovadoras

# Innovación de bienes y servicios ----------------------------------------

# Bienes o servicios nuevos únicamente para su empresa
# (Ya existían en el mercado nacional y/o en el internacional)
print("I1R1C1N (Existían en mercado nacional/internacional - ¿Sí/No?):")
print(EDIT_19_20['I1R1C1N'].value_counts(dropna=False))
print("\nI1R1C2N (Existían en mercado nacional/internacional - Cantidad):")
print(EDIT_19_20['I1R1C2N'].value_counts(dropna=False))

# Bienes o servicios nuevos en el mercado nacional (Ya
# existían en el mercado internacional).
print("\nI1R3C1N (Nuevos en mercado nacional - ¿Sí/No?):")
print(EDIT_19_20['I1R3C1N'].value_counts(dropna=False))
print("\nI1R2C2N (Nuevos en mercado nacional - Cantidad):")
print(EDIT_19_20['I1R2C2N'].value_counts(dropna=False))

# Bienes o servicios nuevos en el mercado internacional.
print("\nI1R3C1N (Nuevos en mercado internacional - ¿Sí/No?):") # This was a typo in R, should be I1R3C1N
print(EDIT_19_20['I1R3C1N'].value_counts(dropna=False))
print("\nI1R3C2N (Nuevos en mercado internacional - Cantidad):")
print(EDIT_19_20['I1R3C2N'].value_counts(dropna=False))

# Validamos que I1R4C2N es la suma de las innovaciones -----
eje = EDIT_19_20[['I1R1C2N', 'I1R2C2N', 'I1R3C2N', 'I1R4C2N']].copy()

# Convert to numeric, replacing 'NOINNO' and NaN with 0
for col in ['I1R1C2N', 'I1R2C2N', 'I1R3C2N', 'I1R4C2N']:
    eje[col] = pd.to_numeric(eje[col], errors='coerce').fillna(0)

eje['innovaciones'] = eje['I1R1C2N'] + eje['I1R2C2N'] + eje['I1R3C2N']

# Check if tables are equal (can be done by comparing value counts or exact equality)
# Note: Value counts might not be exactly equal if there are NaNs differently handled.
# A direct comparison is better for validation.
print("\nValidation of I1R4C2N vs calculated 'innovaciones':")
print((eje['innovaciones'] == eje['I1R4C2N']).all())
print(eje[['innovaciones', 'I1R4C2N']].value_counts()) # This will show how many match

# Terminamos validación

#############################################

# Número total de innovaciones de bienes o servicios nuevos
print("\nNúmero total de innovaciones de bienes o servicios nuevos (I1R4C2N):")
print(EDIT_19_20['I1R4C2N'].value_counts(dropna=False))
#############################################

# Innovación en bienes y servicios mejorados ------------------------------

# Bienes o servicios significativamente mejorados para su empresa
print("\nI1R1C1M (Mejorados para su empresa - ¿Sí/No?):")
print(EDIT_19_20['I1R1C1M'].value_counts(dropna=False))
print("\nI1R1C2M (Mejorados para su empresa - Cantidad):")
print(EDIT_19_20['I1R1C2M'].value_counts(dropna=False))

# Bienes o servicios significativamente mejorados en el mercado nacional
print("\nI1R2C1M (Mejorados en mercado nacional - ¿Sí/No?):")
print(EDIT_19_20['I1R2C1M'].value_counts(dropna=False))
print("\nI1R2C2M (Mejorados en mercado nacional - Cantidad):")
print(EDIT_19_20['I1R2C2M'].value_counts(dropna=False))

# Bienes o servicios significativamente mejorados en el mercado internacional.
print("\nI1R3C1M (Mejorados en mercado internacional - ¿Sí/No?):")
print(EDIT_19_20['I1R3C1M'].value_counts(dropna=False))
print("\nI1R3C2M (Mejorados en mercado internacional - Cantidad):")
print(EDIT_19_20['I1R3C2M'].value_counts(dropna=False))

#############################################
# Total innovaciones de bienes o servicios significativamente mejorados
print("\nTotal innovaciones de bienes o servicios significativamente mejorados (I1R4C2M):")
print(EDIT_19_20['I1R4C2M'].value_counts(dropna=False))
#############################################

# Innovación en métodos de producción -------------------------------------

# mejorados métodos de producción, distribución, entrega
print("\nI1R4C1 (Mejorados métodos de producción, etc. - ¿Sí/No?):")
print(EDIT_19_20['I1R4C1'].value_counts(dropna=False))
print("\nI1R4C2 (Mejorados métodos de producción, etc. - Cantidad):")
print(EDIT_19_20['I1R4C2'].value_counts(dropna=False))

# métodos organizativos implementados en el funcionamiento interno de la empresa
print("\nI1R5C1 (Métodos organizativos - ¿Sí/No?):")
print(EDIT_19_20['I1R5C1'].value_counts(dropna=False))
print("\nI1R5C2 (Métodos organizativos - Cantidad):")
print(EDIT_19_20['I1R5C2'].value_counts(dropna=False))

# nuevas técnicas de comercialización en su empresa
print("\nI1R6C1 (Nuevas técnicas de comercialización - ¿Sí/No?):")
print(EDIT_19_20['I1R6C1'].value_counts(dropna=False))
print("\nI1R6C2 (Nuevas técnicas de comercialización - Cantidad):")
print(EDIT_19_20['I1R6C2'].value_counts(dropna=False))

# Valoración del impacto de la innovación ---------------------------------

# Mejora en la calidad de los bienes
# Nivel de importancia (Alta(1),media(2),nula(3))
print("\nI2R1C1 (Mejora en la calidad de los bienes - Importancia):")
print(EDIT_19_20['I2R1C1'].value_counts(dropna=False))

# Ampliación en la gama de bienes o servicios
print("\nI2R2C1 (Ampliación en la gama de bienes o servicios - Importancia):")
print(EDIT_19_20['I2R2C1'].value_counts(dropna=False))

# Ha mantenido su participación en el mercado geográfico de su empresa
print("\nI2R3C1 (Mantenido participación en el mercado geográfico - Importancia):")
print(EDIT_19_20['I2R3C1'].value_counts(dropna=False))

# Ha ingresado a un mercado geográfico nuevo
print("\nI2R4C1 (Ingresado a un mercado geográfico nuevo - Importancia):")
print(EDIT_19_20['I2R4C1'].value_counts(dropna=False))

# Aumento de la productividad
print("\nI2R5C1 (Aumento de la productividad - Importancia):")
print(EDIT_19_20['I2R5C1'].value_counts(dropna=False))

# Reducción de los costos laborales
print("\nI2R6C1 (Reducción de los costos laborales - Importancia):")
print(EDIT_19_20['I2R6C1'].value_counts(dropna=False))

# Reducción en el uso de materias primas o insumos.
print("\nI2R7C1 (Reducción en el uso de materias primas o insumos - Importancia):")
print(EDIT_19_20['I2R7C1'].value_counts(dropna=False))

# Mejora en el cumplimiento de regulaciones, normas y reglamentos técnicos.
print("\nI2R13C1 (Mejora en el cumplimiento de regulaciones - Importancia):")
print(EDIT_19_20['I2R13C1'].value_counts(dropna=False))

# Porcentaje de impacto ---------------------------------------------------
# pag. 25
print("\nI4R1C1 (Impacto - Ya existían a nivel nal (nal)):")
print(EDIT_19_20['I4R1C1'].value_counts(dropna=False))
print("\nI4R1C2 (Impacto - Ya existían a nivel nal (expor)):")
print(EDIT_19_20['I4R1C2'].value_counts(dropna=False))
print("\nI4R2C1 (Impacto - Productos ya existentes extranj (nal)):")
print(EDIT_19_20['I4R2C1'].value_counts(dropna=False))
print("\nI4R2C2 (Impacto - Productos ya existentes extranj (exporta)):")
print(EDIT_19_20['I4R2C2'].value_counts(dropna=False))
print("\nI4R3C1 (Impacto - Mejoras (ventas nacionales)):")
print(EDIT_19_20['I4R3C1'].value_counts(dropna=False))
print("\nI4R3C2 (Impacto - Mejoras (exportaciones)):")
print(EDIT_19_20['I4R3C2'].value_counts(dropna=False))
print("\nI4R4C1 (Impacto - Productos no innovadores (ventas)):")
print(EDIT_19_20['I4R4C1'].value_counts(dropna=False))
print("\nI4R4C2 (Impacto - Productos no innovadores (exportaciones)):")
print(EDIT_19_20['I4R4C2'].value_counts(dropna=False))

# Resultados económicos ---------------------------------------------------

pd.set_option('display.max_columns', None) # Equivalent to options(scipen=999) for display

# Ventas nacionales totales (miles de pesos corrientes)
print("\nI3R1C1 (Ventas nacionales 2019):")
print(EDIT_19_20['I3R1C1'].head()) # 2019
print("\nI3R2C1 (Ventas nacionales 2020):")
print(EDIT_19_20['I3R2C1'].head()) # 2020

# Convert to numeric, handle 'NOINNO' and NaNs if they exist, then create df_macro
df_macro_2019 = pd.DataFrame({
    'Ventas': pd.to_numeric(EDIT_19_20['I3R1C1'], errors='coerce'),
    'Anio': 2019
}).dropna()

df_macro_2020 = pd.DataFrame({
    'Ventas': pd.to_numeric(EDIT_19_20['I3R2C1'], errors='coerce'),
    'Anio': 2020
}).dropna()

df_macro = pd.concat([df_macro_2019, df_macro_2020])

# Filter out the max value for plotting if it's an outlier, as in the R code
max_sales_2019 = df_macro_2019['Ventas'].max()
df_macro_filtered = df_macro[df_macro['Ventas'] != max_sales_2019]

plt.figure(figsize=(10, 7))
sns.histplot(data=df_macro_filtered, x='Ventas', hue='Anio', kde=True, alpha=0.4, stat='density', common_norm=False)
plt.title('Distribución de Ventas Nacionales por Año (Excluyendo Outlier)')
plt.xlabel('Ventas (miles de pesos corrientes)')
plt.ylabel('Densidad')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

print("\nQuantiles of Ventas:")
print(df_macro['Ventas'].quantile([0.25, 0.5, 0.75]))

print("\nRow with max Ventas:")
max_sales_row = EDIT_19_20.loc[EDIT_19_20['I3R1C1'].idxmax()]
eje = max_sales_row[['NORDEMP', 'CIIU4', 'TIPOLO', 'I3R1C1', 'I3R2C1']].to_frame().T
eje['I3R1C1'] = pd.to_numeric(eje['I3R1C1'], errors='coerce')
eje['I3R2C1'] = pd.to_numeric(eje['I3R2C1'], errors='coerce')
eje['prueba'] = eje['I3R2C1'] - eje['I3R1C1']
print(eje)

# Variables explicativas del modelo ---------------------------------------

# 1. Tamaño de la empresa [TE](personal ocupado)

print("\nMissing values for IV1R11C1 (Personal ocupado 2019):")
print(EDIT_19_20['IV1R11C1'].isna().sum())
print("\nMissing values for IV1R11C2 (Personal ocupado 2020):")
print(EDIT_19_20['IV1R11C2'].isna().sum())

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(EDIT_19_20['IV1R11C1'].dropna(), bins=30, kde=True)
plt.title('Personal Ocupado 2019')
plt.xlabel('Número de Personas')
plt.ylabel('Frecuencia')

plt.subplot(1, 2, 2)
sns.histplot(EDIT_19_20['IV1R11C2'].dropna(), bins=30, kde=True)
plt.title('Personal Ocupado 2020')
plt.xlabel('Número de Personas')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()

# 2. Cualificación del persona (QS, qualifies staff)
# Doctorados, maestria, especialización

# Convert relevant columns to numeric, coercing errors to NaN
for col in ['IV1R1C1', 'IV1R2C1', 'IV1R3C1', 'IV1R11C1',
            'IV1R1C2', 'IV1R2C2', 'IV1R3C2', 'IV1R11C2']:
    EDIT_19_20[col] = pd.to_numeric(EDIT_19_20[col], errors='coerce')

EDIT_19_20['QS_2019'] = 100 * (EDIT_19_20['IV1R1C1'] + EDIT_19_20['IV1R2C1'] + EDIT_19_20['IV1R3C1']) / EDIT_19_20['IV1R11C1']
EDIT_19_20['QS_2020'] = 100 * (EDIT_19_20['IV1R1C2'] + EDIT_19_20['IV1R2C2'] + EDIT_19_20['IV1R3C2']) / EDIT_19_20['IV1R11C2']

print("\nSample of QS_2019 and QS_2020 with missing values:")
print(EDIT_19_20[EDIT_19_20['QS_2020'].isna()][['IV1R11C1', 'IV1R11C2', 'QS_2019', 'QS_2020']].sample(20))

print("\nMissing values for QS_2019:")
print(EDIT_19_20['QS_2019'].isna().sum())
print("\nMissing values for QS_2020 (expected 114, no workers):")
print(EDIT_19_20['QS_2020'].isna().sum())

print("\nLoss of employment (sum of IV1R11C1 - sum of IV1R11C2):")
print(EDIT_19_20['IV1R11C1'].sum() - EDIT_19_20['IV1R11C2'].sum())

# Exportaciones totales (miles de pesos corrientes)
print("\nI3R1C2 (Exportaciones 2019):")
print(EDIT_19_20['I3R1C2'].value_counts(dropna=False))
print("\nI3R2C2 (Exportaciones 2020):")
print(EDIT_19_20['I3R2C2'].value_counts(dropna=False))

# Ventas nacionales totales 2020 (Miles de pesos corrientes)
print("\nI3R2C1 (Ventas nacionales 2020) value counts:")
print(EDIT_19_20['I3R2C1'].value_counts(dropna=False))

# Convert to numeric for histogram
ventas20_numeric = pd.to_numeric(EDIT_19_20['I3R2C1'], errors='coerce').dropna()
plt.figure(figsize=(8, 6))
plt.hist(ventas20_numeric, bins=30, edgecolor='black')
plt.title('Histograma de Ventas Nacionales 2020')
plt.xlabel('Ventas (miles de pesos corrientes)')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()

# Exportaciones totales 2020
print("\nI3R2C2 (Exportaciones 2020) histogram:")
exportaciones20_numeric = pd.to_numeric(EDIT_19_20['I3R2C2'], errors='coerce').dropna()
plt.figure(figsize=(8, 6))
plt.hist(exportaciones20_numeric, bins=30, edgecolor='black')
plt.title('Histograma de Exportaciones Totales 2020')
plt.xlabel('Exportaciones (miles de pesos corrientes)')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()

print("\nMissing values for II1R9C1 (2017):")
print(EDIT_19_20['II1R9C1'].isna().sum()) # 5679 no reportan

df_model21 = EDIT_19_20[EDIT_19_20['II1R10C2'].notna()].copy() # Use .copy() to avoid SettingWithCopyWarning
print("\nCIIU4 distribution in df_model21:")
print(df_model21['CIIU4'].value_counts())

df_model21 = df_model21[['NORDEMP', 'TIPOLO', 'CIIU4', 'IV1R11C2', 'II1R1C2', 'II1R2C2', 'II1R3C2', 'II1R4C2', 'II1R5C2', 'II1R6C2',
                         'II1R7C2', 'II1R8C2', 'II1R9C2', 'III1R3C2']].copy()

nombres = ['NORDEMP', 'TIPOLO', 'CIIU4', 'Personal_18', 'I+D_inversion', 'I+D_adquisicion', 'Maquinaria_adq',
           'Info_teleco', 'Mercadotecnia', 'Transfer_tecno', 'Asis_tecnica', 'Inge_industrial',
           'Formacion', 'Rec_publicos']
df_model21.columns = nombres

print("\nTIPOLO distribution in df_model21 (after filtering):")
print(df_model21['TIPOLO'].value_counts())

print("\nTIPOLO distribution in original EDIT_19_20 (for comparison):")
print(EDIT_19_20['TIPOLO'].value_counts())