import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file all data into a Pandas DataFrame
All_data = pd.read_csv('countries.csv')

# Filter the DataFrame to only include data for Africa
df_africa = All_data[All_data['Continent'] == 'Africa']

# Normalize the input data
df_africa['Population'] = (df_africa['Population'] - df_africa['Population'].min()) / (df_africa['Population'].max() - df_africa['Population'].min())
df_africa['IMF_GDP'] = (df_africa['IMF_GDP'] - df_africa['IMF_GDP'].min()) / (df_africa['IMF_GDP'].max() - df_africa['IMF_GDP'].min())

def loss_function(m, b, points):
    error = 0
    for i in range(len(points)):
        x = points.iloc[i].Population
        y = points.iloc[i].IMF_GDP
        error += (y - (m * x + b))**2
    error / float(len(points))

def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    
    n = len(points)
    
    for i in range(n):
        x = points.iloc[i].Population
        y = points.iloc[i].IMF_GDP
        
        m_gradient += -(2/n) * x *(y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))
        
    m = m_now - L * m_gradient
    b = b_now - L * b_gradient
    
    return m, b 

m = 0
b = 0

L = 0.01
epochs = 1000

for i in range(epochs):
    m, b = gradient_descent(m, b, df_africa, L)

print(m, b)

# Denormalize the data for plotting
x_values = df_africa['Population'].tolist()
x_values = [x * (df_africa['Population'].max() - df_africa['Population'].min()) + df_africa['Population'].min() for x in x_values]
y_values = [m * x + b for x in df_africa['Population']]

plt.scatter(x_values, df_africa['IMF_GDP'],s=5, color='black')
plt.plot(x_values, y_values, color='red',linewidth=0.5)
plt.xlabel('Population')
plt.ylabel('IMF_GDP')
plt.show()
