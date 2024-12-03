import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from app import db
from sqlalchemy import text
import json
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
from collections import defaultdict
import time

def load_data_recommendation_promo():
    transaksi_df = pd.read_sql_query(
        '''
        SELECT * 
        FROM transactions 
        WHERE date >= CURRENT_DATE - INTERVAL '90 days'
        ORDER BY date DESC;
        ''', 
        db.engine
    )
    promo_df = pd.read_sql_query(
        '''
        SELECT * 
        FROM promo;
        ''', 
        db.engine
    )
    
    return transaksi_df, promo_df

def load_data_recommendation_product():
    today = datetime.today()
    first_day_current_month = today.replace(day=1)
    first_day_previous_month = (first_day_current_month - timedelta(days=1)).replace(day=1)
    first_day_three_months_ago = (first_day_previous_month - pd.DateOffset(months=2)).strftime('%Y-%m-%d')
    last_day_previous_month = (first_day_current_month - timedelta(days=1)).strftime('%Y-%m-%d')
    transaksi_df_product = pd.read_sql_query(
        f'''
        SELECT * 
        FROM transactions 
        WHERE date BETWEEN '{first_day_three_months_ago}' AND '{last_day_previous_month}'
        ORDER BY date DESC;
        ''', 
        db.engine
    )

    product = pd.read_sql_query(
        f'''
        SELECT * 
        FROM product
        ''',
        db.engine
    )

    return transaksi_df_product, product


##HYBRID RECOMMENDATION
def train_svd_model(transaksi_df):
    # Create a pivot table (user-item interaction matrix)
    user_item_matrix = transaksi_df.pivot_table(index='cif', columns='merchant_name', aggfunc='size', fill_value=0)

    # Save the columns (merchant names) and index (account IDs) for later use
    merchants = user_item_matrix.columns
    accounts = user_item_matrix.index

    # Convert the interaction matrix to a NumPy array
    user_item_matrix = user_item_matrix.to_numpy(dtype=float)

    # Set the value of k for SVD
    num_accounts, num_merchants = user_item_matrix.shape
    k = min(num_accounts, num_merchants) - 1

    # Apply SVD
    U, sigma, Vt = svds(user_item_matrix, k=k)

    # Convert sigma to a diagonal matrix
    sigma = np.diag(sigma)

    # Reconstruct the approximate interaction matrix
    predicted_matrix = np.dot(np.dot(U, sigma), Vt)

    # Convert the predicted matrix back to a DataFrame for easier interpretation
    predicted_df = pd.DataFrame(predicted_matrix, index=accounts, columns=merchants)

    return predicted_df

def recommend_promos_hybrid(cif, num_recommendations=20):
    # Load data
    transaksi_df, promo_df = load_data_recommendation_promo()

    # Train the SVD model and get predicted scores
    predicted_df = train_svd_model(transaksi_df)

    # Get the predicted scores for the given account
    user_predictions = predicted_df.loc[cif]
    
    # Sort merchants by predicted score
    sorted_merchants = user_predictions.sort_values(ascending=False)
    
    # User's transaction data
    user_transactions = transaksi_df[transaksi_df['cif'] == cif]
    
    # Store recommendations with IDs
    recommendations = []
    for merchant in sorted_merchants.index:
        # Check if there are promos available for this merchant
        merchant_promos = promo_df[promo_df['merchant_name'] == merchant]
        if not merchant_promos.empty:
            recommendations.extend(merchant_promos['id'].tolist())
        else:
            # Check if the user has transactions with this merchant
            user_merchant_transactions = user_transactions[user_transactions['merchant_name'] == merchant]
            if not user_merchant_transactions.empty:
                user_category = user_merchant_transactions['category'].values[0]
                # Check if there are promos available for the same category
                category_promos = promo_df[promo_df['category'] == user_category]
                if not category_promos.empty:
                    recommendations.extend(category_promos['id'].tolist())
        
        # If we've gathered enough recommendations, break the loop
        if len(recommendations) >= num_recommendations:
            break
    
    # If we still don't have enough recommendations, fill with category-based recommendations
    if len(recommendations) < num_recommendations:
        for _, transaction in user_transactions.iterrows():
            category_promos = promo_df[promo_df['category'] == transaction['category']]
            if not category_promos.empty:
                recommendations.extend(category_promos['id'].tolist())
            if len(recommendations) >= num_recommendations:
                break
    
    # Save recommendations to the database
    insertRecommendation(cif, recommendations[:num_recommendations])
    
    return recommendations[:num_recommendations]

# Function to load the pre-trained RNN model
def load_rnn_model(model_filename='model3/01_rnn_recommendation.keras'):
    return load_model(model_filename)

# Preprocess Data
def preprocess_data(transaksi_df):
    le_merchant = LabelEncoder()
    le_cif = LabelEncoder()
    
    transaksi_df['merchant_encoded'] = le_merchant.fit_transform(transaksi_df['merchant_name'])
    transaksi_df['cif_encoded'] = le_cif.fit_transform(transaksi_df['cif'])
    
    return transaksi_df, le_merchant, le_cif

# Function to recommend promos using RNN
def recommend_promos_rnn(cif, model, transaksi_df, promo_df, le_merchant, le_cif, num_recommendations=10):
    try:
        cif_encoded = le_cif.transform([cif])[0]
    except ValueError:
        print(f"CIF {cif} not found in training data. Using default recommendations.")
        return get_default_recommendations(promo_df, num_recommendations)

    user_transactions = transaksi_df[transaksi_df['cif_encoded'] == cif_encoded]
    
    if user_transactions.empty:
        print(f"No transactions found for CIF {cif}. Using default recommendations.")
        return get_default_recommendations(promo_df, num_recommendations)

    user_transactions = user_transactions.sort_values(by='date')
    recent_interactions = user_transactions['merchant_encoded'].values[-10:]
    
    # Pad the sequence if it's shorter than 10
    if len(recent_interactions) < 10:
        recent_interactions = np.pad(recent_interactions, (10 - len(recent_interactions), 0), 'constant')

    # Get the vocabulary size from the model's embedding layer
    vocab_size = model.layers[0].input_dim

    # Clip the merchant encodings to ensure they're within the model's vocabulary size
    recent_interactions = np.clip(recent_interactions, 0, vocab_size - 1)

    try:
        next_merchant_probabilities = model.predict(np.array([recent_interactions]))
    except Exception as e:
        print(f"Error during model prediction: {str(e)}")
        return get_default_recommendations(promo_df, num_recommendations)

    recommended_merchants = np.argsort(next_merchant_probabilities[0])[::-1]
    recommended_merchant_names = le_merchant.inverse_transform(recommended_merchants[:vocab_size])
    
    recommendations = {}
    
    for merchant, score in zip(recommended_merchant_names, next_merchant_probabilities[0][:vocab_size]):
        merchant_promos = promo_df[promo_df['merchant_name'] == merchant]
        
        if not merchant_promos.empty:
            for promo_id in merchant_promos['id'].tolist():
                if promo_id not in recommendations:
                    recommendations[promo_id] = score
        else:
            user_merchant_transactions = user_transactions[user_transactions['merchant_name'] == merchant]
            if not user_merchant_transactions.empty:
                user_category = user_merchant_transactions['category'].values[0]
                category_promos = promo_df[promo_df['category'] == user_category]
                if not category_promos.empty:
                    for promo_id in category_promos['id'].tolist():
                        if promo_id not in recommendations:
                            recommendations[promo_id] = score * 0.5
        
        if len(recommendations) >= num_recommendations:
            break
    
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    promo_ids = [promo_id for promo_id, _ in sorted_recommendations[:num_recommendations]]
    
    return promo_ids

# Function to get default recommendations
def get_default_recommendations(promo_df, num_recommendations=30):
    # You can implement your own logic here, e.g., most popular promos
    return promo_df['id'].tolist()[:num_recommendations]

##PRODUCT
# Function to calculate financial metrics
def calculate_metrics(transaksi_df_product):
    metrics = defaultdict(lambda: {'income': 0, 'expense': 0})
    
    for _, transaction in transaksi_df_product.iterrows():
        cif = transaction['cif']
        amount = transaction['amount']
        category = transaction['category']
        
        if amount > 0:
            metrics[cif]['income'] += amount
        else:
            metrics[cif]['expense'] += abs(amount)
    
    return metrics

def recommend_product(user_metrics, products):
    income = user_metrics['income']
    avg_income = income / 3
    expense = user_metrics['expense']
    avg_expense = expense / 3
    
    recommendations = []
    
    # 1. Kartu Kredit: jika pengeluaran >= 90% dari pendapatan
    if avg_expense >= 0.9 * avg_income:
        recommendations.append(1)
    
    # 2. Deposito, Obligasi, Reksadana: jika expense <= 70% dari income
    if avg_expense <= 0.7 * avg_income:
        recommendations.extend([4, 5, 6])
    
    # 3. BNI Fleksi, BNI Griya, Tabungan Perencanaan, Tabungan Transaksi: 
                    # jika pengeluaran antara 70% dan 90% dari pendapatan
    if 0.7 * avg_income < avg_expense < 0.9 * avg_income:
        recommendations.extend([7, 8, 9, 10])
    
    return sorted(list(set(recommendations)))

# Function to save product recommendations to the database
def insertRecommendation(cif, promo, product):
    promo_ids = list(set(promo)) 
    product_ids = list(set(product)) 
    current_time = datetime.now()

    try:
        with db.engine.connect() as connection:
            # Begin a transaction
            with connection.begin():
                # Check if the CIF exists in the table
                check_query = text("""
                    SELECT EXISTS(
                        SELECT 1 FROM recommendation
                        WHERE cif = :cif
                    )
                """)
                result = connection.execute(check_query, {'cif': cif})
                exists = result.scalar()  # Returns True if CIF exists, False otherwise

                if exists:
                    # CIF exists, update the promo_ids and updated_at timestamp
                    update_query = text("""
                        UPDATE recommendation
                        SET promo_ids = :promo_ids, product_ids = :product_ids, updated_at = :updated_at
                        WHERE cif = :cif
                    """)
                    result = connection.execute(update_query, {
                        'cif': cif,
                        'promo_ids': json.dumps(promo_ids),  
                        'product_ids': json.dumps(product_ids),
                        'updated_at': current_time
                    })
                    print(f"Rows updated: {result.rowcount}")
                else:
                    # CIF does not exist, insert new row with created_at timestamp
                    insert_query = text("""
                        INSERT INTO recommendation (cif, promo_ids, product_ids, created_at)
                        VALUES (:cif, :promo_ids, :product_ids, :created_at)
                    """)
                    result = connection.execute(insert_query, {
                        'cif': cif,
                        'promo_ids': json.dumps(promo_ids),  
                        'product_ids': json.dumps(product_ids),
                        'created_at': current_time
                    })
                    print(f"Rows inserted: {result.rowcount}")
    
    except Exception as e:
        # Handle errors by printing and ensuring proper rollback
        print(f"Error inserting/updating data: {e}")
        raise


