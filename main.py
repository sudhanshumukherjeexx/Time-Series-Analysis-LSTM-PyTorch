import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

st.markdown('## Fetch Rewards - Machine Learning Engineer')
st.markdown('### Take-Home Exercise')
st.markdown('`Live Demo: Every refresh executes new results`')
st.markdown('### GitHub Repo: [🔥](https://github.com/sudhanshumukherjeexx/Time-Series-Analysis-LSTM-PyTorch)')
st.divider()
st.markdown('### Problem Statement')
st.write('At fetch, we are monitoring the number of scanned receipts to our app on a daily base as one of our KPIs. From a business standpoint, we sometimes need to predict the possible number of scanned receipts for a given future month. The data available provides the number of observed scanned receipts each day for the year 2021.')
st.markdown('**Based on this prior knowledge, please develop an algorithm that can predict the approximate number of scanned receipts for each month of 2022.**')
st.divider()

intial_df = pd.read_csv("data_daily.csv")

col1, col2 = st.columns(2)
with col1:
    st.markdown('### Dataset Overview')
    dataframe = pd.read_csv("data_daily.csv")
    st.dataframe(dataframe)
with col2:
    st.markdown('### Shape of Data')
    st.write(dataframe.shape)
    st.markdown('### Features Datatypes')
    st.write(dataframe.dtypes)
    st.markdown('### Missing Value Count')
    st.write(dataframe.isnull().sum())

st.markdown("**To address this problem, let's create more features which we can use in our machine learning model.**")
st.divider()

# Feature Creation and Transformation
st.markdown("### Feature Creature and Transformation")

#function to get season based on Month
def get_season(month):
    if 3 <= month <=5:
        return 0 #Spring
    elif 6 <= month <= 8:
        return 1 #Summer
    elif 9 <= month <= 11:
        return 2 #Fall
    else:
        return 3 #Winter



dataframe['# Date'] = pd.to_datetime(dataframe['# Date'])

# function to generate additional columns to the dataframe
def transform_data(dataframe):
    dataframe['Year'] = dataframe['# Date'].dt.year
    dataframe['Month'] = dataframe['# Date'].dt.month
    dataframe['Day'] = dataframe['# Date'].dt.day
    dataframe['Weekday'] = dataframe['# Date'].dt.weekday
    dataframe['IsWeekend'] = dataframe['# Date'].dt.weekday.isin([5,6]).astype(int)
    dataframe['Season'] = dataframe['Month'].apply(get_season)
    dataframe['WeekOfYear'] = dataframe['# Date'].dt.isocalendar().week.astype(int)
    dataframe['IsMonthStart'] = dataframe['# Date'].dt.is_month_start.astype(int)
    dataframe['IsMonthEnd'] = dataframe['# Date'].dt.is_month_end.astype(int)
    return dataframe

df = transform_data(dataframe)
st.dataframe(df, use_container_width=True)

st.markdown('### New Columns Added')
st.write(df.columns)

# Distribution of Receipt Count (Histogram)
st.markdown("### Distribution of Receipt Count")
fig = px.histogram(df, x='Receipt_Count', title=f'Distribution of Receipt Count')
st.plotly_chart(fig)


# Receipt Count in the year 2021
st.markdown("### Distribution of Receipt Count")
fig = px.line(df, x='# Date', y='Receipt_Count' , title=f'Distribution of Receipt Count')
st.plotly_chart(fig)

# GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
st.markdown(f"### Device Available for Model Training: `{device}`")

# Setting Date as an Index
df.set_index('# Date', inplace=True)

# Features to Numpy
df_numpy = df.to_numpy()
st.markdown('#### Features to Numpy Array')
st.dataframe(df_numpy)
st.write(f"#### Shape of Data: `{df_numpy.shape}`")

# Feature Scaling
scaler = MinMaxScaler(feature_range=(-1,1))
df_numpy = scaler.fit_transform(df_numpy)
st.markdown('#### Feature Scaling')
st.dataframe(df_numpy)

# Train and Test Split
X = df_numpy[:, 1:]
y = df_numpy[:, 0]
st.markdown(f'#### X Shape: `{X.shape}` and y Shape: `{y.shape}`')

X = np.flip(X, axis=1)

split_index = int(len(X) * 0.80)
X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

st.markdown('### Train and Test Split')
st.markdown(f'**X_train Shape:** `{X_train.shape}`, **X_test Shape:** `{X_test.shape}`')
st.markdown(f'**y_train Shape:** `{y_train.shape}`, **y_test Shape:** `{y_test.shape}`')

st.markdown('### Train and Test Split - after reshape')
X_train = X_train.reshape((-1, 9, 1))
X_test = X_test.reshape((-1, 9, 1))

y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

st.markdown(f'**X_train Shape:** `{X_train.shape}`, **X_test Shape:** `{X_test.shape}`')
st.markdown(f'**y_train Shape:** `{y_train.shape}`, **y_test Shape:** `{y_test.shape}`')

# features to tensors
X_train = torch.tensor(X_train.copy()).float()
y_train = torch.tensor(y_train).float()
X_test = torch.tensor(X_test.copy()).float()
y_test = torch.tensor(y_test).float()


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

# Machine Learning Modelling
st.markdown('### LSTM Network - PyTorch')
st.markdown("Train Data: `Year = 2021`")
st.markdown("Test Data:  `Year = 2022`")

st.write(train_dataset)
st.write(test_dataset)

batch_size = 10
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for _, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)
    st.write(x_batch.shape, y_batch.shape)
    break


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
model = LSTM(1, 4, 1)
model.to(device)
st.write(model)


def train_one_epoch():
    model.train(True)
    st.markdown(f"Epoch: {epoch + 1}")
    running_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 10 == 9: #Print every 100 batches
            avg_loss_across_batches = running_loss/100
            st.markdown(f"Batch {batch_index+1}, Loss: {avg_loss_across_batches:.3f}")

            running_loss = 0.0
        st.write()


def validate_one_epoch():
    model.train(False)
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)

    st.markdown(f"Val Loss: {avg_loss_across_batches:.3f}")
    st.divider()
    st.write("")

learning_rate = 0.004
num_epochs = 10
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

columns_per_row = 4
for epoch in range(num_epochs):
    train_one_epoch()
    validate_one_epoch()

with torch.no_grad():
    predicted = model(X_train.to(device)).to('cpu').numpy()

st.markdown("Testing Model on Data Provided")
st.markdown("Model Prediction on Train Data:")
fig, ax = plt.subplots()
ax.plot(y_train, label='Actual Close')
ax.plot(predicted, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()

# Display the plot in Streamlit
st.pyplot(fig)

# Test Model on New Dates
st.markdown('### Test Model for the Year 2022')

new_dates = pd.date_range('2022-01-01', '2022-12-31')
new_dates_df = pd.DataFrame({'# Date': new_dates})

#Transform and Scale Features for New Dates
st.dataframe(new_dates_df)
new_dates_df['Receipt_Count'] = np.zeros((new_dates_df.shape[0], 1))
new_df = transform_data(new_dates_df)
new_df.set_index('# Date', inplace=True)
new_df_numpy = new_df.to_numpy()
new_df_numpy = scaler.transform(new_df_numpy)
test_2022 = new_df_numpy[:, 1:]

#Reshape the Features
new_X = test_2022
new_X_ = np.flip(new_X, axis=1)
new_X_ = new_X_.reshape((-1, 9, 1))
new_X_tensor = torch.tensor(new_X_.copy()).float()

# Testing Model for YEAR - 2022
model.eval()
with torch.no_grad():
    predictions = model(new_X_tensor.to(device))

# Target Variable Interpretation

pred = predictions.cpu()
pred_np = pred.detach().numpy()
pred_reshape = pred_np.reshape(-1,1)
dummy_array = np.zeros((pred_reshape.shape[0], 9))
stacked_array = np.hstack((pred_reshape, dummy_array))
predicted_variable_original = scaler.inverse_transform(stacked_array)[:, 0]
new_dates_df['Predicted Count'] = predicted_variable_original
new_dates_df = new_dates_df.reset_index(drop=False)
predicted_df = new_dates_df[['# Date', 'Predicted Count']]


st.markdown("### 2021")
fig1 = px.line(intial_df, x='# Date', y='Receipt_Count', title=f'Distribution of Receipt Count')
st.plotly_chart(fig1)

st.markdown("### 2022")
fig2 = px.line(predicted_df, x='# Date', y='Predicted Count', title=f'Distribution of Predicted Receipt Count')
st.plotly_chart(fig2)


st.markdown("2020 - Dates & Predicted Count")
st.dataframe(predicted_df)






