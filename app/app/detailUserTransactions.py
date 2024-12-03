from datetime import datetime, timedelta
from app.models import db, Account, Transaction, Recommendation
import json
from sqlalchemy import text
from collections import defaultdict, OrderedDict
from flask import jsonify

def format_amount(amount):
    """Helper function to format amount: no decimal if whole number, otherwise 2 decimal places."""
    if isinstance(amount, float) and amount.is_integer():
        return int(amount)  # If it's a float but has no decimal part, return as int
    elif isinstance(amount, float):
        return round(amount, 2)  # If it's a float with decimals, round to 2 decimal places
    else:
        return amount  # If it's already an int, return it as is

def get_user_details(cif):
    # Fetch the user details by CIF
    user = Account.query.filter_by(cif=cif).first()
    
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    # Fetch total amount for the user's account_number
    total_amount_query = text("""
        SELECT 
            account_number,
            SUM(CASE WHEN status_transaction = 'D' THEN -amount ELSE amount END) AS total_amount
        FROM 
            public.transactions
        WHERE 
            account_number = :account_number
        GROUP BY 
            account_number
    """)
    
    result = db.session.execute(total_amount_query, {'account_number': user.account_number}).fetchone()
    
    # Convert total_amount from Decimal to integer (without decimal)
    total_amount = int(result[1]) if result and result[1] is not None else 0
    
    # Return user details along with the total amount in a JSON response
    return jsonify({
        'user': {
            'cif': user.cif,
            'name': user.name,
            'accountNumber': user.account_number,
            'accountType': user.account_type,
            'totalAmount': total_amount
        }
    }), 200

def get_user_details_and_transactions(cif, start_date, end_date):
    # Fetch the user details by CIF
    user = Account.query.filter_by(cif=cif).first()
    if not user:
        return None

    # Convert start_date and end_date strings to date objects (if they are in string format)
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

    # Ensure that end_date is not earlier than start_date
    if end_date < start_date:
        return {"error": "End date cannot be earlier than start date"}

    # Fetch transactions for the user within the specified date range
    transactions = Transaction.query.filter_by(cif=cif).filter(
        Transaction.date >= start_date,
        Transaction.date <= end_date
    ).all()

    # Separate transactions into income and expenses
    income_transactions = [t for t in transactions if t.status_transaction == 'C']
    expense_transactions = [t for t in transactions if t.status_transaction == 'D']
    
    # Function to get the total by category
    def get_total_by_category(transactions):
        category_totals = defaultdict(float)
        for t in transactions:
            category_totals[t.category] += t.amount
        return category_totals

    # Calculate total income and total expense
    total_income = sum(t.amount for t in income_transactions)
    total_expense = abs(sum(t.amount for t in expense_transactions))

    # Function to calculate percentage per category
    def calculate_percentage(total, amount):
        return round((abs(amount) / abs(total) * 100), 2) if total != 0 else 0

    # Get total amounts per category for income and expense
    income_totals = get_total_by_category(income_transactions)
    expense_totals = get_total_by_category(expense_transactions)

    # Calculate percentages for income and expenses
    income_percentages = [(cat, amount, calculate_percentage(total_income, amount)) for cat, amount in income_totals.items()]
    expense_percentages = [(cat, amount, calculate_percentage(total_expense, amount)) for cat, amount in expense_totals.items()]

    # Sort income and expense by percentage in descending order
    sorted_income_percentages = sorted(income_percentages, key=lambda x: x[2], reverse=True)
    sorted_expense_percentages = sorted(expense_percentages, key=lambda x: x[2], reverse=True)

    # Fetch category names
    category_query = text("""
        SELECT id, name
        FROM public.category
    """)
    category_names = {}
    try:
        with db.engine.connect() as connection:
            result = connection.execute(category_query).fetchall()
            category_names = {row[0]: row[1] for row in result}
    except Exception as e:
        print(f"Error fetching category names: {e}")

    # Format the top income and expense categories with their names, amounts, and percentages
    formatted_top_income = [{
        'category': category_names.get(cat, 'Unknown'), 
        'totalAmount': format_amount(amount), 
        'percentage': percentage
    } for cat, amount, percentage in sorted_income_percentages[:3]]
    
    formatted_top_expense = [{
        'category': category_names.get(cat, 'Unknown'), 
        'totalAmount': abs(format_amount(amount)),
        'percentage': percentage
    } for cat, amount, percentage in sorted_expense_percentages[:3]]

    # Return user details, transactions, and promo details
    return {
        'user': {
            'cif': user.cif,
            'name': user.name,
            'accountNumber': user.account_number,
            'accountType': user.account_type
        },
        'date_range': {
            'startDate': start_date,
            'endDate': end_date
        },
        'totals': {
            'totalIncome': format_amount(total_income),
            'totalExpense': format_amount(total_expense),
            'difference': format_amount(total_income - total_expense)
        },
        'transactions': {
            'topIncomeCategories': formatted_top_income,
            'topExpenseCategories': formatted_top_expense
        },
    }


def get_recommendations(cif, start_date, end_date):

    # Convert start_date and end_date strings to date objects (if they are in string format)
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

    # Ensure that end_date is not earlier than start_date
    if end_date < start_date:
        return {"error": "End date cannot be earlier than start date"}

    # Fetch the user's recommendations based on CIF
    promo_product_recommend = Recommendation.query.filter_by(cif=cif).first()
    
    # Initialize the promo_ids list
    promo_ids = []
    if promo_product_recommend:
        promo_ids = json.loads(promo_product_recommend.promo_ids)

    # Fetch promo details from the promo table based on promo_ids, filtering out invalid promotions
    promo_recommendations = []
    if promo_ids:
        try:
            promo_query = text("""
                SELECT id, category, promo_name, merchant_name, startdate, enddate, promo_url, banner_url
                FROM promo
                WHERE id = ANY(:promo_ids)
                AND (
                    -- If startdate is NULL, check only the enddate
                    (startdate IS NULL AND enddate >= :end_date)
                    OR
                    -- If startdate is not NULL, ensure promo has already started and has not ended
                    (startdate <= :end_date AND enddate >= :end_date)
                )
            """)
            with db.engine.connect() as connection:
                result = connection.execute(promo_query, {
                    'promo_ids': promo_ids,
                    'end_date': end_date  # Use the current date to filter promotions
                }).fetchall()
                promo_recommendations = [OrderedDict([
                                            ('id', row[0]),
                                            ('category', row[1]),
                                            ('promoName', row[2]),
                                            ('merchantName', row[3]),
                                            ('startDate', row[4]),
                                            ('endDate', row[5]),
                                            ('promoUrl', row[6]),
                                            ('bannerUrl', row[7])
                                        ]) for row in result]
        except Exception as e:
            print(f"Error fetching promo details: {e}")

    # Initialize the product_ids list
    product_ids = []
    if promo_product_recommend:
        product_ids = json.loads(promo_product_recommend.product_ids) 

    # Fetch product details from the promo table based on product_ids
    product_recommendations = []
    if product_ids:
        try:
            product_query = text("""
                SELECT id, product_name, product_url, icon_url
                FROM product
                WHERE id = ANY(:product_ids)
            """)
            with db.engine.connect() as connection:
                result = connection.execute(product_query, {'product_ids': product_ids}).fetchall()
                product_recommendations = [{'id': row[0], 
                                  'productName': row[1],
                                  'productUrl': row[2],
                                  'iconUrl': row[3]} for row in result]
        except Exception as e:
            print(f"Error fetching product details: {e}")

    # Fetch category names
    category_query = text("""
        SELECT id, name
        FROM public.category
    """)
    category_names = {}
    try:
        with db.engine.connect() as connection:
            result = connection.execute(category_query).fetchall()
            category_names = {row[0]: row[1] for row in result}
    except Exception as e:
        print(f"Error fetching category names: {e}")

    # Return user details, transactions, and promo details
    return {
        'promo_recommendations': promo_recommendations,
        'product_recommendations': product_recommendations
    }