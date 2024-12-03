from app import db
from datetime import datetime

class Transaction(db.Model):
    __tablename__ = 'transactions'

    id = db.Column(db.String(255), primary_key=True)
    cif = db.Column(db.String(50), nullable=False)  # CIF must not be null as it links to an account
    amount = db.Column(db.Float, nullable=False)
    date = db.Column(db.Date, nullable=False)
    description = db.Column(db.String(255), nullable=False)  # Add description for transaction details
    merchant_name = db.Column(db.String(255))  # Can be null if not relevant
    category = db.Column(db.Integer, nullable=False)  # Refers to a category ID (e.g., Food, Bills, etc.)
    status_transaction = db.Column(db.String(1), nullable=False)  # 'C' for Credit, 'D' for Debit, etc.


class Promo(db.Model):
    __tablename__ = 'promo'
    id = db.Column(db.String(50), primary_key=True)
    promo_name = db.Column(db.String(255))
    category = db.Column(db.Integer)
    merchant_name = db.Column(db.String(255), nullable=True)
    startdate = db.Column(db.Date, nullable=False)
    enddate = db.Column(db.Date, nullable=False)

class Product(db.Model):
    __tablename__ = 'product'
    id = db.Column(db.String(50), primary_key=True)
    product = db.Column(db.String(255))


class Account(db.Model):
    __tablename__ = 'account'

    account_number = db.Column(db.String(15), primary_key=True)
    cif = db.Column(db.String(15), nullable=False, unique=True)
    name = db.Column(db.String(100), nullable=False)
    date_of_birth = db.Column(db.Date, nullable=False)
    username = db.Column(db.String(50), nullable=False, unique=True)
    password = db.Column(db.String(255), nullable=False)
    account_type = db.Column(db.String(50), nullable=False)

class Recommendation(db.Model):
    __tablename__ = 'recommendation'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)  # Serial type
    cif = db.Column(db.String(50), nullable=False)
    promo_ids = db.Column(db.String(255), nullable=False)  # JSON type
    product_ids = db.Column(db.String(50), nullable=False)  # JSON type
    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now, nullable=True)

# class Recommendation(db.Model):
#     __tablename__ = 'recommendation_product_by_transaction'

#     id = db.Column(db.Integer, primary_key=True, autoincrement=True)  # Serial type
#     cif = db.Column(db.String(50), nullable=False)
#     product_ids = db.Column(db.JSON, nullable=False)  # JSON type
#     created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)
#     updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now, nullable=True)

# # models.py
# from flask_sqlalchemy import SQLAlchemy
# from datetime import datetime

# db = SQLAlchemy()

# class Account(db.Model):
#     __tablename__ = 'account'
#     account_number = db.Column(db.String, primary_key=True)
#     cif = db.Column(db.String, unique=True, nullable=False)
#     name = db.Column(db.String, nullable=False)
#     date_of_birth = db.Column(db.Date, nullable=False)
#     username = db.Column(db.String, unique=True, nullable=False)
#     password = db.Column(db.String, nullable=False)

# class Transaction(db.Model):
#     __tablename__ = 'transactions'
#     id = db.Column(db.Integer, primary_key=True, autoincrement=True)
#     cif = db.Column(db.String, db.ForeignKey('account.cif'), nullable=False)
#     amount = db.Column(db.Numeric, nullable=False)
#     date = db.Column(db.Date, nullable=False)
#     description = db.Column(db.String)
#     category = db.Column(db.String)
#     merchant_name = db.Column(db.String)
#     status_transaction = db.Column(db.String, nullable=False)
    
#     # Relationship to Account
#     account = db.relationship('Account', backref=db.backref('transactions', lazy=True))
