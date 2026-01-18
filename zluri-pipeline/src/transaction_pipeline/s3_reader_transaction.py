from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, col, explode, array, lit, when, pow, coalesce, struct
from functools import reduce

# --- 1. CONFIGURATION ---
BUCKET_NAME = "zluri-data-assignment"
BASE_FOLDER = "assignment-jan-2026"
DAY_FOLDER = "sync-day1" 

ENTITY_TRANSACTIONS = "transactions"
ENTITY_CARDS = "cards"
ENTITY_BUDGETS = "budgets"

# Base S3A Path
S3_BASE_PATH = f"s3a://{BUCKET_NAME}/{BASE_FOLDER}"

# --- 2. READER UTILS ---
def read_s3_data(spark, folder_name):
    base_path = f"{S3_BASE_PATH}/{DAY_FOLDER}/{folder_name}/"
    formats = ['json', 'csv', 'parquet']
    found_dfs = []

    print(f"Scanning: {base_path} ...")
    for fmt in formats:
        full_pattern = f"{base_path}*.{fmt}"
        try:
            if fmt == 'json':
                df = spark.read.option("multiline", "true").json(full_pattern)
            elif fmt == 'csv':
                df = spark.read.option("header", "true").option("inferSchema", "true").csv(full_pattern)
            elif fmt == 'parquet':
                df = spark.read.parquet(full_pattern)
            
            if not df.isEmpty():
                print(f"  -> Loaded {fmt} files.")
                found_dfs.append(df)
        except Exception:
            continue

    if not found_dfs:
        print(f"  [!] No data found for entity: {folder_name}")
        return None

    return reduce(lambda df1, df2: df1.unionByName(df2, allowMissingColumns=True), found_dfs)

# --- 3. DATA PROCESSING LOGIC ---

def process_cards_data(spark):
    """ Reads Cards Data for Joins """
    print(f"--- Loading Cards ---")
    df = read_s3_data(spark, ENTITY_CARDS)
    if df is None: return None
    
    if "results" in df.columns:
        df = df.select(explode(col("results")).alias("c")).select("c.*")

    return df.select(
        col("id").alias("card_id"),
        col("name").alias("card_name"),
        col("lastFour").alias("card_last_four"),
        col("type").alias("card_type"),
        col("status").alias("card_status")
    )

def process_budgets_data(spark):
    """ Reads Budgets Data for Joins """
    print(f"--- Loading Budgets ---")
    df = read_s3_data(spark, ENTITY_BUDGETS)
    if df is None: return None
    
    if "results" in df.columns:
        df = df.select(explode(col("results")).alias("b")).select("b.*")
    
    cols = df.columns
    desc_col = col("description").alias("budget_description") if "description" in cols else lit(None).alias("budget_description")

    return df.select(
        col("id").alias("budget_id"),
        col("name").alias("budget_name"),
        desc_col
    )

def process_transactions_schema(spark):
    """
    Extracts Transaction data.
    """
    print(f"--- Loading Transactions ---")
    df_trans_raw = read_s3_data(spark, ENTITY_TRANSACTIONS)
    
    if df_trans_raw is None:
        print("Critical Error: Missing Transaction data.")
        return None

    # FLATTEN DATA
    if "results" in df_trans_raw.columns:
        df_exploded = df_trans_raw.select(explode(col("results")).alias("t"))
    else:
        # --- FIX: Use struct() to wrap columns before aliasing ---
        df_exploded = df_trans_raw.select(struct(col("*")).alias("t"))
    
    cols = df_exploded.select("t.*").columns
    
    def get_col(c_name, alias_name):
        if c_name in cols:
            return col(f"t.{c_name}").alias(alias_name)
        else:
            return lit(None).alias(alias_name)

    # Logic to handle minor units (cents) vs major units (dollars)
    amount_col = (col("t.currencyData.originalCurrencyAmount").cast("double") * \
                  pow(lit(10), -col("t.currencyData.exponent").cast("int"))).alias("original_amount")

    df_final = df_exploded.select(
        col("t.id").alias("transaction_id"),
        get_col("transactionType", "transaction_type"),
        get_col("transactionDate", "transaction_date"),
        get_col("occurredTime", "occurred_time"), 
        get_col("merchantName", "merchant_name"),
        amount_col,
        col("t.currencyData.originalCurrencyCode").alias("currency_code"),
        get_col("cardId", "card_id"),
        get_col("budgetId", "budget_id")
    )
    
    # Date Coalesce Logic
    if "transaction_date" in df_final.columns and "occurred_time" in df_final.columns:
        df_final = df_final.withColumn("transaction_date", coalesce(col("transaction_date"), col("occurred_time")))
        df_final = df_final.drop("occurred_time")
    elif "occurred_time" in df_final.columns:
        df_final = df_final.withColumnRenamed("occurred_time", "transaction_date")
    
    df_final = df_final.dropDuplicates(["transaction_id"])
    
    print(f"  -> Extracted {df_final.count()} unique transactions.")
    return df_final