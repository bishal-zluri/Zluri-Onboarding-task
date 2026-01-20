from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, coalesce, udf, substring, broadcast
from pyspark.sql.types import DoubleType, DecimalType, MapType, StringType
import os
import requests
from dotenv import load_dotenv
import pickle
import time
from datetime import datetime

load_dotenv()

# Imports
from s3_reader_transaction import process_transactions_schema, process_cards_data, process_budgets_data
from transaction_postgres_loader import load_transaction_pipeline, get_existing_transaction_ids, POSTGRES_JAR

# Cache configuration
CACHE_FILE = "fx_rate_cache.pkl"

def load_cache():
    """Load cached exchange rates from disk"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                cache = pickle.load(f)
                return cache
        except Exception as e:
            print(f"[CACHE] Error loading cache: {e}")
    return {}

def save_cache(cache):
    """Save exchange rates to disk cache"""
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache, f)
    except Exception as e:
        print(f"[CACHE] Error saving cache: {e}")

def fetch_exchange_rates_batch(currency_date_pairs):
    """
    Fetch exchange rates for multiple currency-date pairs in batch.
    Uses caching to minimize API calls.
    """
    cache = load_cache()
    rate_map = {}
    api_calls_made = 0
    
    API_KEY = os.getenv("API_KEY")
    
    for currency, date_str in currency_date_pairs:
        if currency == "USD":
            rate_map[f"USD_{date_str}"] = 1.0
            continue
            
        cache_key = f"{currency}_{date_str}"
        
        # Check cache first
        if cache_key in cache:
            rate_map[cache_key] = cache[cache_key]
            continue
        
        # Not in cache, fetch from API
        try:
            url = f"https://api.exchangerate.host/historical?access_key={API_KEY}&date={date_str}"
            
            print(f"[API CALL] Fetching {currency} for {date_str}...")
            response = requests.get(url, timeout=10)
            
            if response.status_code == 429:
                print(f"[ERROR] Rate limit reached. Using fallback 1.0")
                rate_map[cache_key] = 1.0
                cache[cache_key] = 1.0
                continue
            
            data = response.json()
            quotes = data.get("quotes")
            key = f"USD{currency}"
            rate = quotes.get(key) if quotes else None

            if rate is None or float(rate) == 0:
                print(f"[WARN] Missing rate for {key}, using 1.0")
                rate_map[cache_key] = 1.0
                cache[cache_key] = 1.0
                continue

            # Convert USD->XXX to XXX->USD
            converted_rate = 1.0 / float(rate)
            rate_map[cache_key] = converted_rate
            cache[cache_key] = converted_rate
            
            api_calls_made += 1
            if api_calls_made < len(currency_date_pairs):
                time.sleep(0.5) 

        except Exception as e:
            print(f"[ERROR] Exception fetching {currency} on {date_str}: {e}")
            rate_map[cache_key] = 1.0
            cache[cache_key] = 1.0
    
    save_cache(cache)
    return rate_map

def transform_and_load_transactions(spark):
    # 1. Ingest
    df_trans = process_transactions_schema(spark)
    df_cards = process_cards_data(spark)
    df_budgets = process_budgets_data(spark)
    
    if df_trans is None:
        print("No transactions to process.")
        return

    # --- FIX 1: HANDLE NULL AMOUNTS IMMEDIATELY ---
    df_trans = df_trans.withColumn(
        "original_amount", 
        coalesce(col("original_amount"), lit(0))
    )

    # 2. Delta Check
    print("\n=== Checking for New or Updated Transactions (Delta) ===")
    existing_df = get_existing_transaction_ids(spark)
    
    df_new_trans = df_trans
    
    if existing_df and not existing_df.isEmpty():
        existing_df = existing_df.select(
            col("transaction_id").alias("exist_id"), 
            col("original_amount").alias("exist_amount")
        )
        
        df_merged = df_trans.join(
            existing_df, 
            df_trans.transaction_id == existing_df.exist_id, 
            how="left"
        )
        
        df_new_trans = df_merged.filter(
            col("exist_id").isNull() | 
            (col("original_amount").cast("decimal(18,2)") != col("exist_amount").cast("decimal(18,2)"))
        ).select(df_trans.columns)
    
    new_count = df_new_trans.count()
    print(f"  -> Records to Process (New + Updates): {new_count}")
    
    if new_count == 0:
        print("✅ No new or updated transactions found. Pipeline finished.")
        return

    # 3. Currency Conversion
    print("\n=== Converting Currencies to USD ===")
    df_with_date_str = df_new_trans.withColumn(
        "api_date_str", 
        substring(col("transaction_date").cast("string"), 1, 10)
    )
    
    unique_pairs = df_with_date_str.select("currency_code", "api_date_str").distinct().collect()
    currency_date_tuples = [(row["currency_code"], row["api_date_str"]) for row in unique_pairs]
    
    rate_map = fetch_exchange_rates_batch(currency_date_tuples)
    rate_map_broadcast = spark.sparkContext.broadcast(rate_map)
    
    def lookup_rate(currency, date_str):
        cache_key = f"{currency}_{date_str}"
        return rate_map_broadcast.value.get(cache_key, 1.0)
    
    lookup_rate_udf = udf(lookup_rate, DoubleType())
    
    # Apply rates and Calculate USD
    df_final_trans = df_with_date_str.withColumn(
        "exchange_rate", 
        lookup_rate_udf(col("currency_code"), col("api_date_str"))
    ).withColumn(
        "original_amount", 
        col("original_amount").cast("decimal(18, 2)")
    ).withColumn(
        "amount_usd",
        (col("original_amount") * col("exchange_rate")).cast("decimal(18, 2)")
    ).drop("api_date_str")

    # 4. Preparing Dimension DataFrames (Cards & Budgets)
    # Cards Dimension
    df_cards_dim = df_cards.select(
        "card_id", "card_name", "card_last_four", "card_type", "card_status"
    ).distinct()

    # Budgets Dimension
    df_budgets_dim = df_budgets.select(
        "budget_id", "budget_name", "budget_description"
    ).distinct()

    # 5. Preparing Junction Tables (Transaction-Cards, Transaction-Budgets)
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

    # 6. Load
    load_transaction_pipeline(
        spark, 
        df_final_trans, 
        df_trans_cards_join, 
        df_trans_budgets_join,
        df_cards_dim,     # New
        df_budgets_dim    # New
    )

if __name__ == "__main__":
    print(f"--- Launching Transaction Pipeline (Local) ---")
    spark = SparkSession.builder \
        .appName("TransactionTransformer") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.jars", POSTGRES_JAR) \
        .config("spark.driver.extraClassPath", POSTGRES_JAR) \
        .getOrCreate()

    transform_and_load_transactions(spark)