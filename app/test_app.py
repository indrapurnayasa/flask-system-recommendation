from app import db
from sqlalchemy import text

def test_insert():
    try:
        with db.engine.connect() as connection:
            # Insert a simple row for testing
            insert_query = text("""
                INSERT INTO recommendation (cif, promo_ids)
                VALUES (:cif, :promo_ids)
            """)
            
            result = connection.execute(insert_query, {
                'cif': 'test_account',
                'promo_ids': ['1', '2', '3']  # Example promo IDs
            })

            print(f"Rows inserted: {result.rowcount}")
    
    except Exception as e:
        print(f"Error inserting data: {e}")

# Call the test function
test_insert()
