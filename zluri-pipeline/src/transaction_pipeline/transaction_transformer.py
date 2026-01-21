from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, coalesce, udf, substring, broadcast, expr
from pyspark.sql.types import DoubleType, DecimalType, MapType, StringType
import os
import requests
from dotenv import load_dotenv
import time
from datetime import datetime
import urllib3

# Suppress insecure request warnings for local dev
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

# Imports
from s3_reader_transaction import process_transactions_schema, process_cards_data, process_budgets_data
from transaction_postgres_loader import load_transaction_pipeline, get_existing_transaction_ids, POSTGRES_JAR

# Valid Currency List
VALID_CURRENCIES = [
    "USD", "AED", "AFN", "ALL", "AMD", "ANG", "AOA", "ARS", "AUD", "AWG", "AZN", 
    "BAM", "BBD", "BDT", "BGN", "BHD", "BIF", "BMD", "BND", "BOB", "BRL", "BSD", 
    "BTC", "BTN", "BWP", "BYN", "BYR", "BZD", "CAD", "CDF", "CHF", "CLF", "CLP", 
    "CNY", "CNH", "COP", "CRC", "CUC", "CUP", "CVE", "CZK", "DJF", "DKK", "DOP", 
    "DZD", "EGP", "ERN", "ETB", "EUR", "FJD", "FKP", "GBP", "GEL", "GGP", "GHS", 
    "GIP", "GMD", "GNF", "GTQ", "GYD", "HKD", "HNL", "HRK", "HTG", "HUF", "IDR", 
    "ILS", "IMP", "INR", "IQD", "IRR", "ISK", "JEP", "JMD", "JOD", "JPY", "KES", 
    "KGS", "KHR", "KMF", "KPW", "KRW", "KWD", "KYD", "KZT", "LAK", "LBP", "LKR", 
    "LRD", "LSL", "LTL", "LVL", "LYD", "MAD", "MDL", "MGA", "MKD", "MMK", "MNT", 
    "MOP", "MRU", "MUR", "MVR", "MWK", "MXN", "MYR", "MZN", "NAD", "NGN", "NIO", 
    "NOK", "NPR", "NZD", "OMR", "PAB", "PEN", "PGK", "PHP", "PKR", "PLN", "PYG", 
    "QAR", "RON", "RSD", "RUB", "RWF", "SAR", "SBD", "SCR", "SDG", "SEK", "SGD", 
    "SHP", "SLE", "SLL", "SOS", "SRD", "STD", "STN", "SVC", "SYP", "SZL", "THB", 
    "TJS", "TMT", "TND", "TOP", "TRY", "TTD", "TWD", "TZS", "UAH", "UGX", "UYU", 
    "UZS", "VES", "VND", "VUV", "WST", "XAF", "XAG", "XAU", "XCD", "XCG", "XDR", 
    "XOF", "XPF", "YER", "ZAR", "ZMK", "ZMW", "ZWL"
]

def fetch_exchange_rates_batch(currency_date_pairs):
    rate_map = {}
    unique_dates = set(date for _, date in currency_date_pairs)
    
    # Get base URL from env or use default
    base_api_url = os.getenv("API_URL", "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@{date}/v1/currencies/usd.json")

    for date_str in unique_dates:
        # Construct URL dynamically replacing {date}
        if "{date}" in base_api_url:
            url = base_api_url.replace("{date}", date_str)
        else:
            # Fallback if env var is hardcoded or weird
            url = f"https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@{date_str}/v1/currencies/usd.json"
            
        print(f"[API CALL] Fetching rates for {date_str} from {url} ...")
        
        try:
            # verify=False prevents SSL errors common in local environments
            response = requests.get(url, timeout=10, verify=False)
            
            if response.status_code != 200:
                print(f"[ERROR] Failed to fetch rates for {date_str}: Status {response.status_code}")
                rate_map[f"USD_{date_str}"] = 1.0
                continue
                
            data = response.json()
            usd_rates = data.get("usd", {})
            
            # Explicitly set USD base
            rate_map[f"USD_{date_str}"] = 1.0
            
            count_found = 0
            for currency_code, rate_to_usd in usd_rates.items():
                currency_upper = currency_code.upper()
                
                # We only care about currencies present in our VALID list
                if currency_upper in VALID_CURRENCIES and rate_to_usd and float(rate_to_usd) != 0:
                    # API Logic: 1 USD = X Foreign Currency (e.g. 1 USD = 83 INR)
                    # We have Foreign Amount. To get USD: Amount / X
                    # Conversion Rate = 1 / X
                    conversion_rate = 1.0 / float(rate_to_usd)
                    
                    key = f"{currency_upper}_{date_str}"
                    rate_map[key] = conversion_rate
                    count_found += 1
            
            print(f"   -> Successfully cached {count_found} rates for {date_str}.")

        except Exception as e:
            print(f"[ERROR] Exception fetching rates for {date_str}: {e}")
            # Fallback to 1.0 for USD, others will fail/null check
            rate_map[f"USD_{date_str}"] = 1.0

    return rate_map

def transform_and_load_transactions(spark):
    # 1. Ingest
    df_trans = process_transactions_schema(spark)
    df_cards = process_cards_data(spark)
    df_budgets = process_budgets_data(spark)
    
    if df_trans is None:
        print("No transactions to process.")
        return

    # --- VALIDATION STEP ---
    df_clean_trans = df_trans.withColumn("valid_amount", expr("try_cast(original_amount as double)"))
    
    df_valid_trans = df_clean_trans.filter(
        # ID CHECKS
        col("transaction_id").isNotNull() & (col("transaction_id") != "") &
        col("card_id").isNotNull() & (col("card_id") != "") &
        col("budget_id").isNotNull() & (col("budget_id") != "")
    ).filter(
        # AMOUNT CHECK
        col("valid_amount").isNotNull() & (col("valid_amount") >= 0)
    ).filter(
        # CURRENCY CHECK
        col("currency_code").isin(VALID_CURRENCIES)
    ).drop("valid_amount")

    valid_count = df_valid_trans.count()
    if valid_count == 0:
        print("No valid transactions remaining after validation.")
        return

    # 2. Delta Check
    print("\n=== Checking for New or Updated Transactions (Delta) ===")
    existing_df = get_existing_transaction_ids(spark)
    
    df_new_trans = df_valid_trans
    
    if existing_df and not existing_df.isEmpty():
        existing_df = existing_df.select(
            col("transaction_id").alias("exist_id"), 
            col("original_amount").alias("exist_amount")
        )
        
        df_merged = df_valid_trans.join(
            existing_df, 
            df_valid_trans.transaction_id == existing_df.exist_id, 
            how="left"
        )
        
        # Filter for new or changed amounts
        df_new_trans = df_merged.filter(
            col("exist_id").isNull() | 
            (col("original_amount").cast("decimal(18,2)") != col("exist_amount").cast("decimal(18,2)"))
        ).select(df_valid_trans.columns)
    
    count_to_process = df_new_trans.count()
    print(f"  -> Records to Process (New + Updates): {count_to_process}")
    
    if count_to_process == 0:
        print("✅ No new or updated transactions found. Pipeline finished.")
        return

    # 3. Currency Conversion
    print("\n=== Converting Currencies to USD ===")
    # Extract date string (YYYY-MM-DD) from transaction_date
    df_with_date_str = df_new_trans.withColumn(
        "api_date_str", 
        substring(col("transaction_date").cast("string"), 1, 10)
    )
    
    # Identify unique (Currency, Date) pairs
    unique_pairs = df_with_date_str.select("currency_code", "api_date_str").distinct().collect()
    currency_date_tuples = [(row["currency_code"], row["api_date_str"]) for row in unique_pairs]
    
    # Fetch rates from API (No Cache, with SSL verify=False)
    rate_map = fetch_exchange_rates_batch(currency_date_tuples)
    rate_map_broadcast = spark.sparkContext.broadcast(rate_map)
    
    def lookup_rate(currency, date_str):
        cache_key = f"{currency}_{date_str}"
        # Default to 1.0 only if USD, otherwise could use a sentinel to detect failure
        return rate_map_broadcast.value.get(cache_key, 1.0)
    
    lookup_rate_udf = udf(lookup_rate, DoubleType())
    
    # Apply Rates
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

    # 4. Preparing Dimension DataFrames
    valid_card_ids = [row.card_id for row in df_final_trans.select("card_id").distinct().collect()]
    valid_budget_ids = [row.budget_id for row in df_final_trans.select("budget_id").distinct().collect()]

    df_cards_dim = df_cards.filter(col("card_id").isin(valid_card_ids)).select(
        "card_id", "card_name", "card_last_four", "card_type", "card_status"
    ).distinct()

    df_budgets_dim = df_budgets.filter(col("budget_id").isin(valid_budget_ids)).select(
        "budget_id", "budget_name", "budget_description"
    ).distinct()

    # 5. Junction Tables
    df_trans_cards_join = df_final_trans.alias("t").join(
        df_cards_dim.alias("c"),
        col("t.card_id") == col("c.card_id"),
        how="inner" 
    ).select(
        col("t.transaction_id"), col("c.card_id"), col("c.card_name"),
        col("c.card_last_four"), col("c.card_type"), col("c.card_status")
    )
    
    df_trans_budgets_join = df_final_trans.alias("t").join(
        df_budgets_dim.alias("b"),
        col("t.budget_id") == col("b.budget_id"),
        how="inner" 
    ).select(
        col("t.transaction_id"), col("b.budget_id"), 
        col("b.budget_name"), col("b.budget_description")
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