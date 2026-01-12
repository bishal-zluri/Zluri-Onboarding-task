from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, col, explode
from functools import reduce

# --- 1. CONFIGURATION ---
BUCKET_NAME = "zluri-data-assignment"
BASE_FOLDER = "assignment-jan-2026"
DAY_FOLDER = "sync-day1" 

ENTITY_TRANSACTIONS = "transactions"

# Base S3A Path
S3_BASE_PATH = f"s3a://{BUCKET_NAME}/{BASE_FOLDER}"

# --- 2. SPARK SESSION FACTORY ---
def create_spark_session(app_name="TransactionSchemaExtractor"):
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider") \
        .config("spark.hadoop.fs.s3a.threads.keepalivetime", "60") \
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "30000") \
        .config("spark.hadoop.fs.s3a.connection.timeout", "30000") \
        .config("spark.hadoop.fs.s3a.multipart.purge.age", "86400") \
        .getOrCreate()

# --- 3. GENERIC READER FUNCTION ---
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

# --- 4. DATA PROCESSING LOGIC ---

def process_transactions_schema(spark):
    """
    Extracts Transaction data and maps it strictly to the provided Schema Diagram.
    Target Columns: transaction_id, transaction_type, amount_in_dollars, original_amount, card_id
    """
    
    # 1. Load Raw Data
    df_trans_raw = read_s3_data(spark, ENTITY_TRANSACTIONS)
    
    if df_trans_raw is None:
        print("Critical Error: Missing Transaction data.")
        return

    # 2. FLATTEN DATA
    # The JSON structure is { "results": [ ... ] }, so we must explode the array first.
    df_exploded = df_trans_raw.select(explode(col("results")).alias("t"))
    
    # 3. SELECT & MAP COLUMNS
    # We map the JSON fields to the 'Transactions' entity attributes from your diagram.
    df_final = df_exploded.select(
        col("t.id").alias("transaction_id"),                                # Mapped to schema PK
        col("t.transactionType").alias("transaction_type"),       
        
        # Accessing nested field for original amount
        col("t.currencyData.originalCurrencyAmount").alias("original_amount"), 
        
        # Accessing the currency_code
        col("t.currencyData.originalCurrencyCode").alias("currency_code"),

        # Accessing the card id
        col("t.cardId").alias("card_id")                                    # Foreign Key to Card table
    )
    
    print(f"\n--- extracted {df_final.count()} transactions matching schema ---")
    
    return df_final

# --- 5. MAIN EXECUTION ---
if __name__ == "__main__":
    spark_session = create_spark_session()
    final_df = process_transactions_schema(spark_session)
    
    if final_df:
        final_df.printSchema()
        final_df.show(truncate=False)