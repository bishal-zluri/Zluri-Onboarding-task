from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, coalesce, udf, substring
from pyspark.sql.types import DoubleType, DecimalType
import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Imports
from s3_reader_transaction import process_transactions_schema, process_cards_data, process_budgets_data
from transaction_postgres_loader import load_transaction_pipeline, get_existing_transaction_ids, POSTGRES_JAR

def transform_and_load_transactions(spark):
    # 1. Ingest
    df_trans = process_transactions_schema(spark)
    df_cards = process_cards_data(spark)
    df_budgets = process_budgets_data(spark)
    
    if df_trans is None:
        print("No transactions to process.")
        return

    # 2. Delta Check with Update Logic
    print("\n=== Checking for New or Updated Transactions (Delta) ===")
    existing_df = get_existing_transaction_ids(spark)
    
    df_new_trans = df_trans
    
    if existing_df and not existing_df.isEmpty():
        # Rename columns to avoid ambiguity during join
        existing_df = existing_df.select(
            col("transaction_id").alias("exist_id"), 
            col("original_amount").alias("exist_amount")
        )
        
        # Left join to identify New or Updated records
        # Logic: 
        # 1. exist_id is Null -> New Record
        # 2. exist_id is Not Null BUT original_amount != exist_amount -> Updated Record
        df_merged = df_trans.join(
            existing_df, 
            df_trans.transaction_id == existing_df.exist_id, 
            how="left"
        )
        
        df_new_trans = df_merged.filter(
            col("exist_id").isNull() | 
            (col("original_amount").cast("decimal(18,2)") != col("exist_amount").cast("decimal(18,2)"))
        ).select(df_trans.columns) # Keep only original columns
    
    new_count = df_new_trans.count()
    print(f"  -> Records to Process (New + Updates): {new_count}")
    
    if new_count == 0:
        print("✅ No new or updated transactions found. Pipeline finished.")
        return

    # 3. Currency Conversion
    print("\n=== Converting Currencies to USD ===")

    # --- UDF DEFINITION ---
    # UPDATED: Removed caching of default 1.0 failure.
    # If the API fails, it returns 1.0 but does NOT store it in `_cache`.
    # This ensures that on the next run, it tries the API again instead of using the bad cached value.
    def fetch_exchange_rate(currency, date_str):

        if currency == "USD":
            return 1.0

        try:
            API_KEY = os.getenv("API_KEY")
            url = (
                "http://api.currencylayer.com/historical"
                f"?access_key={API_KEY}&date={date_str}&source=USD"
            )

            response = requests.get(url, timeout=5)
            data = response.json()

            if not data.get("success"):
                raise ValueError("Currency API returned success=false")

            quotes = data.get("quotes")
            if not quotes:
                raise ValueError("Missing quotes in API response")

            key = f"USD{currency}"
            rate = quotes.get(key)

            if rate is None or float(rate) == 0:
                raise ValueError(f"Missing rate for {key}")

            # USD -> GBP → invert to get GBP -> USD
            return 1.0 / float(rate)

        except Exception as e:
            print(f"[WARN] FX fallback used for {currency} on {date_str}: {e}")
            return 1.0

 
    convert_currency_udf = udf(fetch_exchange_rate, DoubleType())
    
    # Extract YYYY-MM-DD
    df_with_date_str = df_new_trans.withColumn(
        "api_date_str", 
        substring(col("transaction_date").cast("string"), 1, 10)
    )
    
    # Calculate amount_usd
    df_final_trans = df_with_date_str.withColumn(
        "exchange_rate", 
        convert_currency_udf(col("currency_code"), col("api_date_str"))
    ).withColumn(
        "original_amount", 
        col("original_amount").cast("decimal(18, 2)")
    ).withColumn(
        "amount_usd",
        (col("original_amount") * col("exchange_rate")).cast("decimal(18, 2)")
    ).drop("api_date_str")

    # 4. Joins
    print("\n=== Building Join Tables ===")

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