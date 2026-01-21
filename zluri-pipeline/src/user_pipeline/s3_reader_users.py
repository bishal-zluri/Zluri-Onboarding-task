from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, col, explode_outer, lit, coalesce, concat_ws
from functools import reduce
import os

# --- CONFIGURATION ---
DAY_FOLDER = "sync-day1"

ENTITY_AGENTS_INDEX = "agents"          
ENTITY_AGENT_DETAILS = "agent_details"  
ENTITY_ROLES = "roles"                  

def create_spark_session(app_name="AgentDataExtractor"):
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.driver.host", "127.0.0.1") \
        .getOrCreate()

# --- LOCAL READER UTILS ---
def read_local_data(spark, folder_name):
    # 1. Get current directory (e.g., .../src/user_pipeline)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 2. Navigate up 3 levels to reach 'ONBOARDING TASK' root
    project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
    # 3. Construct path
    target_dir = os.path.join(project_root, DAY_FOLDER, folder_name)
    base_path = f"file://{target_dir}"
    
    formats = ['json', 'csv', 'parquet']
    found_dfs = []
    
    print(f"Scanning Local Path: {base_path} ...")
    
    for fmt in formats:
        full_pattern = os.path.join(base_path, f"*.{fmt}")
        try:
            if not os.path.exists(target_dir):
                continue
            
            if fmt == 'json':
                df = spark.read.option("multiline", "true").json(full_pattern)
            elif fmt == 'csv':
                df = spark.read.option("header", "true").option("inferSchema", "true").csv(full_pattern)
            elif fmt == 'parquet':
                df = spark.read.parquet(full_pattern)
                
            df = df.withColumn("source_path", input_file_name())
            if not df.isEmpty():
                print(f"  -> Loaded {fmt} files.")
                found_dfs.append(df)
        except Exception:
            continue
            
    if not found_dfs:
        print(f"  [!] No data found for entity: {folder_name} at {target_dir}")
        return None
        
    return reduce(lambda df1, df2: df1.unionByName(df2, allowMissingColumns=True), found_dfs)

def process_agents_data(spark):
    """
    Returns the joined dataframe. 
    Implements FALLBACK logic: If agent_details is missing, use info from agents index.
    """
    df_idx = read_local_data(spark, ENTITY_AGENTS_INDEX)
    df_det = read_local_data(spark, ENTITY_AGENT_DETAILS)
    df_rol = read_local_data(spark, ENTITY_ROLES)

    if not df_idx: 
        print("Required Agent Index missing.")
        return None
    
    # Check if we actually have details
    has_details = df_det is not None
    
    if not has_details:
        print("⚠️ Agent Details missing. Using Agent Index data only.")
        # Create a true dummy for the join to syntactically work
        # We select "id" so the join condition 'on="id"' works
        df_det = df_idx.select("id").limit(0).alias("det") 
    
    # Join Users + Details
    df_users = df_idx.alias("idx").join(df_det.alias("det"), on="id", how="left")

    # --- FALLBACK COLUMN LOGIC ---
    idx_cols = df_idx.columns
    
    # Name Fallback: Check 'name' or composite 'first_name + last_name' in index
    if "name" in idx_cols:
        fallback_name = col("idx.name")
    elif "first_name" in idx_cols and "last_name" in idx_cols:
        fallback_name = concat_ws(" ", col("idx.first_name"), col("idx.last_name"))
    else:
        fallback_name = lit("Unknown Name")

    # Other Fallbacks
    fallback_email = col("idx.email") if "email" in idx_cols else lit("no-email@placeholder.com")
    fallback_created = col("idx.created_at") if "created_at" in idx_cols else lit(None)
    fallback_updated = col("idx.updated_at") if "updated_at" in idx_cols else lit(None)

    # --- SAFE SELECTION ---
    # Only try to access 'det.contact.name' if we know df_det was loaded.
    # Otherwise use lit(None).
    
    det_name = col("det.contact.name") if has_details else lit(None)
    det_email = col("det.contact.email") if has_details else lit(None)
    det_created = col("det.created_at") if has_details else lit(None)
    det_updated = col("det.updated_at") if has_details else lit(None)
    det_group_ids = col("det.group_ids") if has_details else lit(None)
    
    # FIX: Cast to array<string> so explode_outer doesn't crash on NULL
    det_role_ids = col("det.role_ids") if has_details else lit(None).cast("array<string>")

    # --- SELECTION WITH COALESCE ---
    # coalesce(A, B) returns A if not null, otherwise B.
    # This prioritizes Agent Details (A), then falls back to Agent Index (B).
    
    df_exploded = df_users.select(
        col("idx.id").alias("user_id"),
        coalesce(det_name, fallback_name).alias("user_name"),
        coalesce(det_email, fallback_email).alias("user_email"),
        coalesce(det_created, fallback_created).alias("created_at"),
        coalesce(det_updated, fallback_updated).alias("updated_at"),
        det_group_ids.alias("raw_group_ids"), 
        explode_outer(det_role_ids).alias("role_id")
    )

    # Join with Roles to get Role Names
    if df_rol:
        df_final = df_exploded.alias("u").join(df_rol.alias("r"), col("u.role_id") == col("r.id"), how="left") \
            .select(
                col("u.user_id"), col("u.user_name"), col("u.user_email"), 
                col("u.created_at"), col("u.updated_at"), col("u.role_id"), 
                col("r.name").alias("role_name"), col("r.description").alias("role_desc")
            )
    else:
        df_final = df_exploded.select("*", lit(None).alias("role_name"), lit(None).alias("role_desc"))

    return df_final

if __name__ == "__main__":
    print("--- Running User Reader Individually ---")
    spark = create_spark_session()
    try:
        df = process_agents_data(spark)
        if df:
            df.show(truncate=False)
            print(f"Total User Records (Exploded): {df.count()}")
    finally:
        spark.stop()