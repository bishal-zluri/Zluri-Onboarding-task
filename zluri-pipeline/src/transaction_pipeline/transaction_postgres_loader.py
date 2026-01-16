import os
from pyspark.sql.functions import col
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
POSTGRES_JAR = "/Users/bishalpb/onboarding task/jars/postgresql-42.7.8.jar"
DB_URL = "jdbc:postgresql://localhost:5432/postgres"

DB_PROPERTIES = {
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASS", "password"),
    "driver": "org.postgresql.Driver"
}

# Tables
TABLE_TRANS = "transactions"
TABLE_TRANS_CARDS = "transaction_cards"
TABLE_TRANS_BUDGETS = "transaction_budgets"

# Stages
TEMP_TRANS = "trans_stage"
TEMP_TC = "trans_cards_stage"
TEMP_TB = "trans_budgets_stage"

def execute_raw_sql(spark, sql_query):
    conn = None
    try:
        driver_manager = spark.sparkContext._gateway.jvm.java.sql.DriverManager
        conn = driver_manager.getConnection(DB_URL, DB_PROPERTIES["user"], DB_PROPERTIES["password"])
        conn.setAutoCommit(True) 
        stmt = conn.createStatement()
        stmt.execute(sql_query)
    except Exception as e:
        print(f"   [SQL Execute Info]: {str(e)[:100]}...")
        raise e
    finally:
        if conn: conn.close()

def _add_column_if_missing(spark, table, col_def):
    try:
        # print(f"   -> checking column: {col_def} on {table}")
        sql = f"ALTER TABLE {table} ADD COLUMN {col_def}"
        execute_raw_sql(spark, sql)
    except Exception as e:
        if "already exists" in str(e).lower():
            pass
        else:
            print(f"⚠️ Warning during schema update: {e}")

def init_db(spark):
    print("\n--- [DB] Initializing Transaction Tables ---")
    
    # 1. Transactions Table
    try:
        execute_raw_sql(spark, f"""
        CREATE TABLE IF NOT EXISTS {TABLE_TRANS} (
            transaction_id TEXT PRIMARY KEY,
            transaction_type TEXT,
            transaction_date DATE,
            original_amount DECIMAL(18, 2),
            currency_code TEXT,
            amount_usd DECIMAL(18, 2),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
    except Exception as e:
        if "already exists" not in str(e).lower(): raise e

    _add_column_if_missing(spark, TABLE_TRANS, "merchant_name TEXT")
    _add_column_if_missing(spark, TABLE_TRANS, "card_id TEXT")
    _add_column_if_missing(spark, TABLE_TRANS, "budget_id TEXT")
    
    # 2. Transaction Cards
    try:
        execute_raw_sql(spark, f"""
        CREATE TABLE IF NOT EXISTS {TABLE_TRANS_CARDS} (
            transaction_id TEXT,
            card_id TEXT,
            card_name TEXT,
            card_last_four TEXT,
            card_type TEXT,
            card_status TEXT,
            PRIMARY KEY (transaction_id, card_id)
        );
        """)
    except Exception as e:
        if "already exists" not in str(e).lower(): raise e
    
    # 3. Transaction Budgets
    try:
        execute_raw_sql(spark, f"""
        CREATE TABLE IF NOT EXISTS {TABLE_TRANS_BUDGETS} (
            transaction_id TEXT,
            budget_id TEXT,
            budget_name TEXT,
            budget_description TEXT,
            PRIMARY KEY (transaction_id, budget_id)
        );
        """)
    except Exception as e:
        if "already exists" not in str(e).lower(): raise e

def get_existing_transaction_ids(spark):
    """ 
    Used for Delta Load filtering.
    UPDATED: Returns both transaction_id and original_amount to detect updates.
    """
    try:
        init_db(spark)
        df = spark.read.jdbc(url=DB_URL, table=TABLE_TRANS, properties=DB_PROPERTIES)
        # Select both ID and Amount to check for value changes
        return df.select("transaction_id", "original_amount")
    except Exception:
        return None

def load_transaction_pipeline(spark, df_trans, df_cards_join, df_budgets_join):
    init_db(spark)
    
    # --- 1. Load Transactions (Upsert) ---
    if not df_trans.isEmpty():
        print(f"--- Loading {df_trans.count()} Transactions ---")
        df_trans.write.jdbc(DB_URL, TEMP_TRANS, "overwrite", DB_PROPERTIES)
        
        upsert_sql = f"""
        INSERT INTO {TABLE_TRANS} (
            transaction_id, transaction_type, transaction_date,
            original_amount, currency_code, amount_usd, merchant_name,
            card_id, budget_id
        )
        SELECT 
            transaction_id, transaction_type, transaction_date::date,
            original_amount, currency_code, amount_usd, merchant_name,
            card_id, budget_id
        FROM {TEMP_TRANS}
        ON CONFLICT (transaction_id) 
        DO UPDATE SET 
            amount_usd = EXCLUDED.amount_usd,
            original_amount = EXCLUDED.original_amount,
            merchant_name = EXCLUDED.merchant_name,
            card_id = EXCLUDED.card_id,
            budget_id = EXCLUDED.budget_id,
            transaction_date = EXCLUDED.transaction_date;
        """
        execute_raw_sql(spark, upsert_sql)
        print("✅ Transactions Loaded.")

    # --- 2. Load Transaction-Cards ---
    if not df_cards_join.isEmpty():
        print(f"--- Loading {df_cards_join.count()} Card Links ---")
        df_cards_join.write.jdbc(DB_URL, TEMP_TC, "overwrite", DB_PROPERTIES)
        
        insert_sql = f"""
        INSERT INTO {TABLE_TRANS_CARDS} (
            transaction_id, card_id, card_name, 
            card_last_four, card_type, card_status
        )
        SELECT 
            transaction_id, card_id, card_name, 
            card_last_four, card_type, card_status 
        FROM {TEMP_TC}
        ON CONFLICT (transaction_id, card_id) 
        DO UPDATE SET
            card_name = EXCLUDED.card_name,
            card_last_four = EXCLUDED.card_last_four,
            card_type = EXCLUDED.card_type,
            card_status = EXCLUDED.card_status;
        """
        execute_raw_sql(spark, insert_sql)
        print("✅ Transaction-Card Links Loaded.")

    # --- 3. Load Transaction-Budgets ---
    if not df_budgets_join.isEmpty():
        print(f"--- Loading {df_budgets_join.count()} Budget Links ---")
        df_budgets_join.write.jdbc(DB_URL, TEMP_TB, "overwrite", DB_PROPERTIES)
        
        insert_sql = f"""
        INSERT INTO {TABLE_TRANS_BUDGETS} (
            transaction_id, budget_id, budget_name, budget_description
        )
        SELECT 
            transaction_id, budget_id, budget_name, budget_description 
        FROM {TEMP_TB}
        ON CONFLICT (transaction_id, budget_id)
        DO UPDATE SET
            budget_name = EXCLUDED.budget_name,
            budget_description = EXCLUDED.budget_description;
        """
        execute_raw_sql(spark, insert_sql)
        print("✅ Transaction-Budget Links Loaded.")
        
    try:
        execute_raw_sql(spark, f"DROP TABLE IF EXISTS {TEMP_TRANS}")
        execute_raw_sql(spark, f"DROP TABLE IF EXISTS {TEMP_TC}")
        execute_raw_sql(spark, f"DROP TABLE IF EXISTS {TEMP_TB}")
    except:
        pass