import streamlit as st
import time
import numpy as np

st.set_page_config(page_title="Anomaly Detection", page_icon="📈")

st.markdown("# Anomaly Detection")
st.sidebar.header("Anomaly Detection")


# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# Imports
import re
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import itertools
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('ggplot')
#%matplotlib inline
import plotly.express as px
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
#import lightning as pl
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint

import lightning.pytorch as pl
import torch.nn.functional as F

pd.options.mode.chained_assignment = None  # default='warn'



# Data Ingestion
uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
#df = pd.read_csv(uploaded_files)
#df = pd.read_csv('/Users/aaronjawan/Downloads/Tool_Sensor_Data.csv')
for uploaded_file in uploaded_files:
    # Progress bar
    progress_bar = st.progress(0)
    
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(uploaded_file)

# To decide whether missing data is Missing Completely At Random (MCAR),
# If it is, we can discard it


    def pipeline_0(df, threshold_percentage):
        # Step 1: Identify columns with missing values more than specified threshold
        # Missing Completely At Random (MCAR), hence discard missing columns
        # Calculate the threshold
        threshold = len(df) * threshold_percentage
        columns_to_remove = df.columns[df.isnull().sum() > threshold]


        # Step 2: Identify columns with only one unique value
        # No statistical significance
        columns_with_one_unique_value = df.columns[df.nunique() <= 1]


        # Step 3: Remove identified columns
        # Combine both sets of columns to be removed
        columns_to_remove = columns_to_remove.union(columns_with_one_unique_value)
        removed_columns = list(columns_to_remove)

        # Drop the identified columns from the dataframe
        df_processed = df.drop(columns=columns_to_remove)


        # Step 4: Remove rows with any missing values
        # Missing Completely At Random (MCAR), hence discard missing rows
        df_processed = df_processed.dropna()

        return removed_columns, df_processed

    removed_columns, df = pipeline_0(df, 0.5)
    print("Removed columns:", removed_columns)




    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    def pipeline_1(df, timestamp_cols, test_size, random_state, mode, preprocessor=None):
        """
        Preprocesses the input DataFrame by dropping timestamp columns, scaling numerical features,
        one-hot encoding categorical features, and preparing training data.

        Parameters:
        - df (pd.DataFrame): The input DataFrame to be preprocessed.
        - timestamp_cols (list): List of columns to drop related to timestamps. Default is ['TimeStamp', 'RunStartTime'].
        - test_size (float): The proportion of the dataset to include in the test split. Default is 0.2.
        - random_state (int): Random seed for reproducibility. Default is 42.
        - mode (str): Mode of operation, either 'train' or 'inference'. Default is 'train'.
        - preprocessor (ColumnTransformer): Fitted ColumnTransformer instance for 'inference' mode. Default is None.

        Returns:
        - X_train (pd.DataFrame): Preprocessed training feature data.
        - all_feature_names (list): List of feature names after one-hot encoding.
        - X_val_df (pd.DataFrame): Preprocessed validation feature data (if X_test is provided).
        """
        # Drop timestamp columns
        # The models we use (LSTM AutoEncoders) to detect anomalies in the time-series data will use the sequence of the data as the temporal feature, not the timestamp
        df = df.drop(columns=timestamp_cols)

        # Define columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        if mode == 'train':
            # Split data before transformations to prevent data leakage
            X_train, X_val = train_test_split(df, test_size=test_size, random_state=random_state)

            # Preprocessing
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_cols),
                    ('cat', OneHotEncoder(), categorical_cols)
                ])

            # Prepare data
            X_train = preprocessor.fit_transform(X_train)
            X_val = preprocessor.transform(X_val)

            # Get feature names after OneHotEncoder transformation
            onehot_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
            all_feature_names = numerical_cols + onehot_feature_names.tolist()

            # Convert csr_matrix to DataFrame
            X_train = pd.DataFrame(X_train, columns=all_feature_names)
            X_val = pd.DataFrame(X_val, columns=all_feature_names)

            return X_train, X_val, all_feature_names, preprocessor

        elif mode == 'inference':
            if preprocessor is None:
                raise ValueError("For inference mode, a fitted preprocessor must be provided.")

            # Prepare data
            X_inference = preprocessor.transform(df)

            # Get feature names after OneHotEncoder transformation
            onehot_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
            all_feature_names = numerical_cols + onehot_feature_names.tolist()

            # Convert csr_matrix to DataFrame
            X_inference = pd.DataFrame(X_inference, columns=all_feature_names)

            return X_inference, all_feature_names

        else:
            raise ValueError("Invalid mode. Please use 'train' or 'inference'.")

    # Pipeline 1
    # As this is unsupervised learning (as we have no labels), we would want to use all of our dataset as our training set
    X_train, X_val, all_feature_names, preprocessor = pipeline_1(df, ['TimeStamp', 'RunStartTime'], test_size=0.00001, random_state=42, mode='train')
    print(X_train.shape)
    print(X_val.shape)
    print(all_feature_names)



    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from scipy.sparse import csr_matrix

    X_train_selected = X_train.copy()

    # Initialize and fit the Isolation Forest model
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_forest.fit(X_train_selected)

    # Predict anomalies
    anomalies = iso_forest.predict(X_train_selected)

    # Convert anomalies to a DataFrame and add it to X_train_selected
    X_train_selected['anomaly'] = anomalies

    # Create a synthetic target variable
    X_train_selected['target'] = (X_train_selected['anomaly'] == -1).astype(int)

    # Fit a Random Forest model to estimate feature importance
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_selected.drop(columns=['anomaly', 'target']), X_train_selected['target'])

    # Get feature importances from the Random Forest model
    importances = rf_model.feature_importances_

    # Create a DataFrame of features and their importances
    feature_importance_df = pd.DataFrame({
        'feature': X_train_selected.drop(columns=['anomaly', 'target']).columns,
        'importance': importances
    })

    # Sort features by importance
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    # Select the top N important features (e.g., top 10 features)
    top_n = 15
    top_features = feature_importance_df.head(top_n)['feature'].tolist()

    # Create a new DataFrame with only the most important features
    X_train = X_train_selected[top_features]
    print(feature_importance_df.head(top_n))
    print(X_train.shape)



    def pipeline_2(df, batch_size, seq_len):
        """
        Prepares data for LSTM model using the given X_train and df.

        Parameters:
        X_train (pd.DataFrame): The input training data.
        df (pd.DataFrame): The dataframe containing the "Run" column for batch size and sequence length calculations.

        Returns:
        DataLoader: A DataLoader object containing the prepared data.
        """
        print("X_train shape: ", X_train.shape)

        # batch size and sequence length for LSTM
        print("batch_size: ", batch_size)

        print("seq_len: ", seq_len)

        # Step 1: Convert the dataframe to a dense numpy array
        dense_matrix = X_train.to_numpy()
        print("Step 1 Convert the dataframe to a dense numpy array")

        # Step 2: Convert the dense numpy array to a PyTorch tensor
        tensor_data = torch.tensor(dense_matrix, dtype=torch.float32)
        print("Step 2 Convert the dense numpy array to a PyTorch tensor: ", tensor_data.shape)

        # Step 3: Prepare the data for LSTM (assuming each row is a sequence)
        # Create sequences using a sliding window
        sequences = []
        for i in range(tensor_data.shape[0] - seq_len + 1):
            sequences.append(tensor_data[i:i + seq_len])

        sequences = torch.stack(sequences)
        print("Step 3 Create sequences using a sliding window")
        print("No of samples: ", sequences.shape[0])
        print("Length of each sequence: ", sequences.shape[1])
        print("No of features: ", sequences.shape[2])

        # Step 4: Create a TensorDataset and DataLoader
        dataset = TensorDataset(sequences)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        print("Step 4 Create a TensorDataset and DataLoader")

        return dataloader, batch_size, seq_len
    
    progress_bar.progress(25, text="Preprocessing Done")

    # Define batch size and sequence length
    batch_size = X_train.shape[0] // df['Run'].nunique()
    seq_len = X_train.shape[0] // df['Run'].nunique()

    dataloader, batch_size, seq_len = pipeline_2(X_train, batch_size, seq_len)


    class VariationalLSTMAutoencoder(pl.LightningModule):
        def __init__(self, input_dim, hidden_dim, latent_dim, seq_len, num_layers=2, lr=1e-3):
            super(VariationalLSTMAutoencoder, self).__init__()
            self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

            self.fc_mu = nn.Linear(hidden_dim, latent_dim) #a fully connected layer that maps the hidden state to the mean (mu) of the latent distribution
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim) #a fully connected layer that maps the hidden state to the log variance (logvar) of the latent distribution

            self.decoder = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True)
            self.fc_decoder = nn.Linear(hidden_dim, input_dim) #a fully connected layer that maps the LSTM output to the final decoded sequence

            self.hidden_dim = hidden_dim
            self.latent_dim = latent_dim
            self.seq_len = seq_len
            self.lr = lr

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(self, x):
            # Encoder
            _, (hidden, _) = self.encoder(x)
            hidden = hidden[-1]  # Take the last layer's hidden state

            mu = self.fc_mu(hidden)
            logvar = self.fc_logvar(hidden)
            z = self.reparameterize(mu, logvar)

            # Repeat latent vector for each time step in the sequence
            z = z.unsqueeze(1).repeat(1, self.seq_len, 1)

            # Decoder
            decoded, _ = self.decoder(z)
            decoded = self.fc_decoder(decoded)
            return decoded, mu, logvar

        def training_step(self, batch, batch_idx):
            # Unpack the input data from the batch. The batch contains a single tensor, which is the input sequence.
            x, = batch

            # Pass the input sequence through the model to get the reconstructed sequence, mean, and log variance of the latent space.
            x_hat, mu, logvar = self(x)

            # Calculate the reconstruction loss using Mean Squared Error (MSE) between the original and reconstructed sequences.
            recon_loss = F.mse_loss(x_hat, x, reduction='mean')

            # Calculate the Kullback-Leibler Divergence (KLD) loss, which measures the difference between the learned distribution and a standard normal distribution.
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # Combine the reconstruction loss, KLD loss to get the total loss.
            loss = recon_loss + kld_loss

            # Log the total training loss for monitoring and visualization purposes.
            self.log('train_loss', loss)
            return loss

        def validation_step(self, batch, batch_idx):
            x, = batch
            x_hat, mu, logvar = self(x)
            recon_loss = F.mse_loss(x_hat, x, reduction='mean')
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kld_loss
            self.log('val_loss', loss)
            tensorboard_logs = {'val_loss': loss}
            return {"val_loss": loss}

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            return optimizer

    # Usage
    num_features = X_train.shape[1]
    seq_len = 63
    lstm_vae = VariationalLSTMAutoencoder(input_dim=num_features, hidden_dim=32, latent_dim=10, seq_len=seq_len)

    # LSTM VariationalAutoEncoder
    # Load the trained model
    lstm_vae = torch.load("lstm_vae.pt")

    progress_bar.progress(50, text="Detecting Anomalies")


    def compute_anomaly_threshold(dataloader, model, param):
        # Define the loss function
        criterion = nn.MSELoss()

        # Switch model to evaluation mode
        model.eval()

        # Compute reconstruction errors
        reconstruction_errors = []
        with torch.no_grad():
            for batch in dataloader:
                sequences_batch = batch[0]
                if param == 'lstm_ae':
                    output = model(sequences_batch)
                elif param == 'lstm_vae':
                    output, _, _ = model(sequences_batch)  # Unpack the output tuple
                else:
                    raise ValueError("Invalid param value. Must be 'lstm_ae' or 'lstm_vae'.")
                loss = criterion(output, sequences_batch)
                # Collect the loss values
                reconstruction_errors.append(loss.item())

        # Convert list to a NumPy array for easier manipulation
        reconstruction_errors = np.array(reconstruction_errors)

        # Calculate the threshold for anomaly detection
        threshold = np.mean(reconstruction_errors) + 3 * np.std(reconstruction_errors)
        print(f'Threshold for anomaly detection: {threshold}')
        return threshold
    
    

    def detect_anomalies(dataloader, model, threshold, param):
        # Define the loss function
        criterion = nn.MSELoss()

        anomalies = []
        anomaly_indices = []
        anomaly_scores = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                sequences_batch = batch[0]
                for seq_idx, sequence in enumerate(sequences_batch):
                    sequence = sequence.unsqueeze(0)  # Add batch dimension

                    if param == 'lstm_ae':
                        output = model(sequence)
                    elif param == 'lstm_vae':
                        output, _, _ = model(sequence)  # Unpack the output tuple, taking only the first element (reconstruction)
                    else:
                        raise ValueError("Invalid parameter. Use 'lstm_ae' or 'lstm_vae'.")

                    loss = criterion(output, sequence)
                    if loss.item() > threshold:
                        anomalies.append(sequence.numpy())  # Collect anomalies
                        anomaly_indices.append(batch_idx * len(sequences_batch) + seq_idx)  # Collect anomaly indices
                        anomaly_scores.append(loss.item())  # Collect anomaly scores

        print(f'Number of anomalies detected: {len(anomalies)}')
        print(f'Indices of anomalies: {anomaly_indices}')
        return anomalies, anomaly_indices, anomaly_scores

    # Calculate Threshold
    #threshold = compute_anomaly_threshold(dataloader, lstm_vae, 'lstm_vae')

    # Detect Anomalies
    anomalies, anomaly_indices, anomaly_scores = detect_anomalies(dataloader, lstm_vae, 1.7537, 'lstm_vae')


    # Filter the DataFrame to include only the rows corresponding to the anomaly indices
    anomalous_df = df.iloc[anomaly_indices]

    # Add the anomaly scores as a new column
    anomalous_df['anomaly_score'] = anomaly_scores
    anomalous_df_sorted = anomalous_df.sort_values(by='anomaly_score', ascending=False)

    progress_bar.progress(75, text="Visualizing Anomalies")

    # Apply t-SNE
    tsne = TSNE(n_components=1, perplexity=30, n_iter=250, random_state=42, verbose=2)
    Y_tsne = tsne.fit_transform(X_train)

    # Create the DataFrame
    tsne_df = pd.DataFrame(Y_tsne, columns=['1d representation'])
    tsne_df['Is_Anomaly'] = tsne_df.index.isin(anomaly_indices)
    tsne_df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    tsne_df['LOT_ID'] = df['LOT_ID']

    # Add the index as a column to be used in the tooltip
    tsne_df['Index'] = tsne_df.index

    # Define a custom color map
    color_map = {0: 'blue', 1: 'red'}  # Assuming 0 is normal and 1 is anomaly

    # Create the scatter plot
    fig = px.scatter(tsne_df, x='TimeStamp', y='1d representation', color='Is_Anomaly', color_discrete_map=color_map, opacity=0.6, hover_data=['Index', 'LOT_ID'])
    fig.update_layout(title_text='1D Representation of Dataset With Respect to Time (With Flagged Anomalies)')
    #fig.show()
    st.plotly_chart(fig)




    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    # Filter for only anomalies
    filtered_df = tsne_df[tsne_df['Is_Anomaly']]
    # Ensure anomaly_scores is a 2D array
    anomaly_scores_fit = np.array(anomaly_scores).reshape(-1, 1)
    # Scale the anomaly scores
    filtered_df['anomaly_scores'] = scaler.fit_transform(anomaly_scores_fit)

    # Create the scatter plot
    fig = px.scatter(filtered_df, x='TimeStamp', y='1d representation', color='anomaly_scores',
                    size='anomaly_scores', opacity=0.6, hover_data=['Index', 'LOT_ID', 'anomaly_scores'])

    fig.update_layout(title_text='1D Representation of Dataset With Respect to Time (With Only Anomalies)')
    #fig.show()
    st.plotly_chart(fig)

    progress_bar.progress(100, text="Done")
