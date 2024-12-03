from flask import Blueprint, jsonify, request, session
from app.recommendation import load_data_recommendation_promo, load_data_recommendation_product, preprocess_data, load_rnn_model, recommend_promos_rnn, calculate_metrics, recommend_product, insertRecommendation
from app.detailUserTransactions import get_user_details_and_transactions, get_user_details, get_recommendations
from app.models import Transaction, Account
import time

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return 'Welcome to Kompre!'

@main.route('/user', methods=['GET'])
def show_user_details():
    try:
        # Get the query parameter 'cif' from the request
        cif = request.args.get('cif')

        if not cif:
            return jsonify({'error': 'cif parameter is required'}), 400

        # Call the function with CIF as a query parameter
        result = get_user_details(cif)
        
        return result

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Helper function to convert SQLAlchemy model to dictionary
def object_as_dict(obj):
    return {column.name: getattr(obj, column.name) for column in obj.__table__.columns}

# Example query to retrieve all transactions
@main.route('/transactions', methods=['GET'])
def get_transactions():
    transactions = Transaction.query.all()
    # Convert transactions to a list of dictionaries
    transactions_dict = [object_as_dict(t) for t in transactions]
    return jsonify(transactions=transactions_dict)

# @main.route('/recommendations', methods=['POST'])
# def recommendations_hybrid():
#     try:
#         # Extract JSON data from request body
#         data = request.get_json()
        
#         print(f"Received data: {data}")  # Debug line to check the received data
        
#         # Get 'cif' from the received JSON data
#         cif = data.get('cif')
        
#         if not cif:
#             return jsonify({'error': 'cif is required'}), 400
        
#         # Call the recommendation logic with the cif
#         recommended_promos = recommend_promos_hybrid(cif)
        
#         # Return the recommendations as a JSON response
#         return jsonify({
#             'cif': cif,
#             'recommended_promos': recommended_promos
#         }), 200
    
#     except Exception as e:
#         # Handle errors (e.g., recommendation logic failure, bad input)
#         return jsonify({'error': str(e)}), 500

@main.route('/recommendations/rnn_model', methods=['POST'])
def recommendations_rnn():
    try:
        data = request.get_json()
        print(f"Received data: {data}")
        
        cif = data.get('cif')
        if not cif:
            return jsonify({'error': 'cif is required'}), 400
        
        transaksi_df, promo_df = load_data_recommendation_promo()  # You need to implement this function to load your data
        transaksi_df, le_merchant, le_cif = preprocess_data(transaksi_df)
        model_filename = 'rnn_model.keras'
        rnn_model = load_rnn_model(model_filename)
        
        recommended_promos = recommend_promos_rnn(
            cif=cif,
            model=rnn_model,
            transaksi_df=transaksi_df,
            promo_df=promo_df,
            le_merchant=le_merchant,
            le_cif=le_cif,
            num_recommendations=30
        )
        
        return jsonify({
            'cif': cif,
            'recommended_promos': recommended_promos
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    
# API endpoint for recommendations
# @main.route('/recommendations/transaction_based', methods=['POST'])
# def recommendations_transaction_based():
#     try:
#         data = request.get_json()
#         print(f"Received data: {data}")
        
#         cif = data.get('cif')
#         if not cif:
#             return jsonify({'error': 'cif is required'}), 400
        
#         transactions, products = load_data()
#         user_metrics = calculate_metrics(transactions)
        
#         if cif not in user_metrics:
#             return jsonify({'error': 'CIF not found in transaction data'}), 404
        
#         recommended_products = recommend_product(user_metrics[cif], products)
        
#         # Save recommendations to the database
#         saveProductRecommendations(cif, recommended_products)
        
#         return jsonify({
#             'cif': cif,
#             'recommended_products': recommended_products
#         }), 200
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500   

# API endpoint for recommendations
@main.route('/recommendations', methods=['POST'])
def recommendations():
    try:
        start_time = time.time()
        data = request.get_json()
        
        cif = data.get('cif')
        if not cif:
            return jsonify({'error': 'cif is required'}), 400
        
        transaksi_df_product, products = load_data_recommendation_product()
        user_metrics = calculate_metrics(transaksi_df_product)
        

        transaksi_df, promo_df = load_data_recommendation_promo()
        transaksi_df, le_merchant, le_cif = preprocess_data(transaksi_df)
        
        model_filename = 'model3/01_rnn_recommendation.keras'
        rnn_model = load_rnn_model(model_filename)
        
        recommended_promos = recommend_promos_rnn(
            cif=cif,
            model=rnn_model,
            transaksi_df=transaksi_df,
            promo_df=promo_df,
            le_merchant=le_merchant,
            le_cif=le_cif,
            num_recommendations=20
        )
        
        if cif not in user_metrics:
            return jsonify({'error': 'CIF not found in transaction data'}), 404
        
        recommended_products = recommend_product(user_metrics[cif], products)
        
        total_elapsed_time = time.time() - start_time
        print(f"Total inference time: {total_elapsed_time:.2f} seconds")

        # Save recommendations to the database
        insertRecommendation(cif, recommended_promos, recommended_products)
        
        return jsonify({
            'cif': cif,
            'promo_recommendation': recommended_promos,
            'product_recommendation': recommended_products
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500    


@main.route('/detailTransaction', methods=['GET'])
def user_detail_transaction():
    try:
        # Get the query parameters (cif, start_date, and end_date)
        cif = request.args.get('cif')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        if not cif:
            return jsonify({'error': 'cif is required'}), 400
        
        if not start_date or not end_date:
            return jsonify({'error': 'start_date and end_date are required'}), 400

        # Call the function from detailUserTransactions.py with start_date and end_date
        result = get_user_details_and_transactions(cif, start_date, end_date)
        
        if result:
            return jsonify(result), 200
        else:
            return jsonify({'error': 'User not found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@main.route('/detailRecommendation', methods=['GET'])
def user_detail_recommendation():
    try:
        # Get the query parameters (cif, start_date, and end_date)
        cif = request.args.get('cif')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        if not cif:
            return jsonify({'error': 'cif is required'}), 400
        
        if not start_date or not end_date:
            return jsonify({'error': 'start_date and end_date are required'}), 400

        # Call the function from detailUserTransactions.py with start_date and end_date
        result = get_recommendations(cif, start_date, end_date)
        
        if result:
            return jsonify(result), 200
        else:
            return jsonify({'error': 'User not found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main.route('/login', methods=['POST'])
def login():
    # Get JSON data from the request
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    # Check if both username and password were provided
    if not username or not password:
        return jsonify({"message": "Username and password required"}), 400

    # Query the account by username
    account = Account.query.filter_by(username=username).first()

    # If the account does not exist, return an error
    if not account:
        return jsonify({"message": "Invalid username or password"}), 401

    # Verify the password (assuming it's hashed)
    if not (account.password, password):
        return jsonify({"message": "Invalid username or password"}), 401

    # If login is successful, create a session or return a token
    # session['account_number'] = account.account_number  # Start a session
    return jsonify({"message": "Login successful", "name": account.name, "cif":account.cif}), 200

@main.route('/logout', methods=['POST'])
def logout():

    # Return a response indicating successful logout
    return jsonify({"message": "Logout successful"}), 200