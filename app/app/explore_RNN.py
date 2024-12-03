import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from app.models import db, Transaction, Promo

def load_data():
    # Load transaction data from the database into a DataFrame
    transaksi_df = pd.read_sql_query('SELECT * FROM transactions', Transaction)
    
    # Load promo data from the database into a DataFrame
    promo_df = pd.read_sql_query('SELECT * FROM promo', P)
    
    return transaksi_df, promo_df

def preprocess_data(transaksi_df):
    # Encoding the 'merchant_name' and 'cif' to numerical values
    le_merchant = LabelEncoder()
    le_cif = LabelEncoder()
    
    transaksi_df['merchant_encoded'] = le_merchant.fit_transform(transaksi_df['merchant_name'])
    transaksi_df['cif_encoded'] = le_cif.fit_transform(transaksi_df['cif'])
    
    return transaksi_df, le_merchant, le_cif

def prepare_sequences(transaksi_df, sequence_length=10):
    # Grouping transactions by 'cif' and sorting them by 'date' (assuming it exists)
    transaksi_df = transaksi_df.sort_values(by=['cif', 'date'])
    
    sequences = []
    targets = []
    
    for cif, group in transaksi_df.groupby('cif_encoded'):
        merchant_list = group['merchant_encoded'].tolist()
        
        # Create sequences for the RNN
        for i in range(len(merchant_list) - sequence_length):
            sequences.append(merchant_list[i:i + sequence_length])
            targets.append(merchant_list[i + sequence_length])
    
    return np.array(sequences), np.array(targets)

def buildModelRNN(num_merchants, sequence_length=10, embedding_dim=50):
    model = Sequential()
    model.add(Embedding(input_dim=num_merchants, output_dim=embedding_dim, input_length=sequence_length))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(num_merchants, activation='softmax'))  # Output layer
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_rnn_model(sequences, targets, num_merchants):
    # Splitting the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(sequences, targets, test_size=0.2, random_state=42)
    
    # Build the model
    model = buildModelRNN(num_merchants=num_merchants)
    
    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    
    return model

def recommend_promos_rnn(cif, model, transaksi_df, le_merchant, le_cif, num_recommendations=20):
    # Get the merchant interaction sequence for the given cif
    cif_encoded = le_cif.transform([cif])[0]
    user_transactions = transaksi_df[transaksi_df['cif_encoded'] == cif_encoded]
    
    # Sort user's transactions and extract recent interactions
    user_transactions = user_transactions.sort_values(by='date')
    recent_interactions = user_transactions['merchant_encoded'].values[-10:]  # Last 10 merchants
    
    # Predict the next merchant the user is likely to interact with
    next_merchant_probabilities = model.predict(np.array([recent_interactions]))
    
    # Get the top merchants to recommend
    recommended_merchants = np.argsort(next_merchant_probabilities[0])[::-1][:num_recommendations]
    
    # Decode merchant indices back to merchant names
    recommended_merchant_names = le_merchant.inverse_transform(recommended_merchants)
    
    return recommended_merchant_names

# Example usage:
transaksi_df, promo_df = load_data()
transaksi_df, le_merchant, le_cif = preprocess_data(transaksi_df)
sequences, targets = prepare_sequences(transaksi_df)

# Get the number of unique merchants for embedding
num_merchants = len(le_merchant.classes_)

# Train the RNN model
rnn_model = train_rnn_model(sequences, targets, num_merchants)

# Get recommendations for a specific user
recommendations = recommend_promos_rnn(cif="2024474209", model=rnn_model, transaksi_df=transaksi_df, le_merchant=le_merchant, le_cif=le_cif)
print(recommendations)
