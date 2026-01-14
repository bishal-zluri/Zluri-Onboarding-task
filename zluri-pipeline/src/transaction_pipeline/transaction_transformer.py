from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, coalesce, udf
from pyspark.sql.types import DoubleType, DecimalType
import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Imports
from s3_reader_transaction import process_transactions_schema, process_cards_data, process_budgets_data
from transaction_postgres_loader import load_transaction_pipeline, get_existing_transaction_ids, POSTGRES_JAR

# --- CURRENCY CONVERSION LOGIC ---
def fetch_exchange_rate(currency, date_str):
    if currency == "USD":
        return 1.0
    try:
        # FIXED: Hardcoded Key to ensure workers can access it
        API_KEY = os.getenv("API_KEY")
        url = f"https://v6.exchangerate-api.com/v6/{API_KEY}/latest/{currency}"
        
        response = requests.get(url, timeout=3)
        data = response.json()
        
        # Check success and get rate
        if data.get("result") == "success":
            rate = data.get("conversion_rates", {}).get("USD")
            return float(rate) if rate else 1.0
        return 1.0
        
    except Exception:
        # In production, log this failure
        return 1.0

convert_currency_udf = udf(fetch_exchange_rate, DoubleType())

def transform_and_load_transactions(spark):
    # 1. Ingest
    df_trans = process_transactions_schema(spark)
    df_cards = process_cards_data(spark)
    df_budgets = process_budgets_data(spark)
    
    if df_trans is None:
        print("No transactions to process.")
        return

    # 2. Delta Check
    print("\n=== Checking for New Transactions (Delta) ===")
    existing_ids = get_existing_transaction_ids(spark)
    
    df_new_trans = df_trans
    if existing_ids and not existing_ids.isEmpty():
        df_new_trans = df_trans.join(existing_ids, on="transaction_id", how="left_anti")
    
    new_count = df_new_trans.count()
    print(f"  -> New Records: {new_count}")
    
    if new_count == 0:
        print("✅ No new transactions found. Pipeline finished.")
        return

    # 3. Currency Conversion
    print("\n=== Converting Currencies to USD ===")
    
    # We apply UDF. Original amount is already decimal-corrected by Reader.
    df_with_usd = df_new_trans.withColumn(
        "exchange_rate", 
        convert_currency_udf(col("currency_code"), col("transaction_date").cast("string"))
    ).withColumn(
        "original_amount", 
        col("original_amount").cast("decimal(18, 2)")
    )
    
    df_final_trans = df_with_usd.withColumn(
        "amount_usd",
        col("original_amount") * col("exchange_rate")
    )

    # 4. Joins (Expanded columns)
    print("\n=== Building Join Tables ===")

    # A. Transaction-Cards
    df_trans_cards_join = df_final_trans.alias("t").join(
        df_cards.alias("c"),
        col("t.card_id") == col("c.card_id"),
        how="inner" 
    ).select(
        col("t.transaction_id"),
        col("c.card_id"),
        col("c.card_name"),
        col("c.card_last_four"),
        col("c.card_type"),
        col("c.card_status")
    )
    
    # B. Transaction-Budgets
    df_trans_budgets_join = df_final_trans.alias("t").join(
        df_budgets.alias("b"),
        col("t.budget_id") == col("b.budget_id"),
        how="inner" 
    ).select(
        col("t.transaction_id"),
        col("b.budget_id"),
        col("b.budget_name"),
        col("b.budget_description")
    )

    # 5. Load
    load_transaction_pipeline(
        spark, 
        df_final_trans, 
        df_trans_cards_join, 
        df_trans_budgets_join
    )

if __name__ == "__main__":
    print(f"--- Launching Transaction Pipeline ---")
    spark = SparkSession.builder \
        .appName("TransactionTransformer") \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider") \
        .config("spark.hadoop.fs.s3a.threads.keepalivetime", "60") \
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "30000") \
        .config("spark.hadoop.fs.s3a.connection.timeout", "30000") \
        .config("spark.hadoop.fs.s3a.multipart.purge.age", "86400") \
        .config("spark.jars", POSTGRES_JAR) \
        .config("spark.driver.extraClassPath", POSTGRES_JAR) \
        .getOrCreate()

    transform_and_load_transactions(spark)