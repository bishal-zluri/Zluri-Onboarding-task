from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
# 1. IMPORT THE RUNNER
from prefect.task_runners import ConcurrentTaskRunner 
from datetime import timedelta
import sys
import os

# --- PATH CONFIGURATION ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

sys.path.insert(0, os.path.join(CURRENT_DIR, 'user_pipeline'))
sys.path.insert(0, os.path.join(CURRENT_DIR, 'groups_pipeline'))
sys.path.insert(0, os.path.join(CURRENT_DIR, 'transaction_pipeline'))

from user_pipeline.user_transformer import transform_and_reconcile_users, POSTGRES_JAR
from groups_pipeline.groups_transformer import transform_and_reconcile_groups
from transaction_pipeline.transaction_transformer import transform_and_load_transactions
from pyspark.sql import SparkSession

# --- SPARK SESSION MANAGEMENT ---
@task(name="Initialize Spark", cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def get_spark_session(app_name="Prefect-ETL"):
    return SparkSession.builder \
        .appName(app_name) \
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

# --- ENTITY TASKS (Same as before) ---
@task(name="Run User Pipeline", retries=2, retry_delay_seconds=60)
def run_user_pipeline(spark_session, sync_day):
    logger = get_run_logger()
    logger.info(f"Starting USER pipeline for {sync_day}...")
    try:
        transform_and_reconcile_users(spark_session)
        logger.info(f"USER pipeline completed successfully for {sync_day}.")
    except Exception as e:
        logger.error(f"USER pipeline failed: {e}")
        raise e

@task(name="Run Group Pipeline", retries=2, retry_delay_seconds=60)
def run_group_pipeline(spark_session, sync_day):
    logger = get_run_logger()
    logger.info(f"Starting GROUP pipeline for {sync_day}...")
    try:
        transform_and_reconcile_groups(spark_session)
        logger.info(f"GROUP pipeline completed successfully for {sync_day}.")
    except Exception as e:
        logger.error(f"GROUP pipeline failed: {e}")
        raise e

@task(name="Run Transaction Pipeline", retries=2, retry_delay_seconds=60)
def run_transaction_pipeline(spark_session, sync_day):
    logger = get_run_logger()
    logger.info(f"Starting TRANSACTION pipeline for {sync_day}...")
    try:
        transform_and_load_transactions(spark_session)
        logger.info(f"TRANSACTION pipeline completed successfully for {sync_day}.")
    except Exception as e:
        logger.error(f"TRANSACTION pipeline failed: {e}")
        raise e

@task(name="Mark Inactive Users")
def post_sync_user_(spark_session):
    get_run_logger().info("User status reconciliation complete.")

@task(name="Mark Inactive Groups")
def post_sync_group_(spark_session):
    get_run_logger().info("Group hierarchy status propagation complete.")

# --- MAIN FLOW ---

# 2. CONFIGURE THE RUNNER IN THE DECORATOR
# ... (imports remain the same) ...

@flow(name="Zluri Data Assignment Pipeline", task_runner=ConcurrentTaskRunner())
def main_pipeline_flow():
    # ... (setup remains the same) ...
    
    # --- PARALLEL EXECUTION BLOCK ---
    logger = get_run_logger()
    logger.info("Starting Main Orchestrator...")
    spark = get_spark_session()

    day = "sync-day2"
    logger.info(f"=== PROCESSING: {day} ===")

    import s3_reader_users as ur
    ur.DAY_FOLDER = day
    import s3_reader_groups as gr
    gr.DAY_FOLDER = day
    import s3_reader_transaction as tr
    tr.DAY_FOLDER = day

    # 1. Submit Roots
    user_future = run_user_pipeline.submit(spark, day)
    transaction_future = run_transaction_pipeline.submit(spark, day)
    
    # 2. Submit Dependents
    group_future = run_group_pipeline.submit(spark, day, wait_for=[user_future])
    
    # 3. Submit Final Steps (Leaf Nodes)
    # Capture these futures so we can wait for them!
    post_user_future = post_sync_user_.submit(spark, wait_for=[user_future])
    post_group_future = post_sync_group_.submit(spark, wait_for=[group_future])
    
    logger.info(f"All pipelines submitted. Waiting for completion...")

    # --- CRITICAL FIX: WAIT FOR TASKS ---
    # We wait for the end of every independent chain.
    
    transaction_future.wait()   # Wait for Transaction Chain
    post_user_future.wait()     # Wait for User Chain
    post_group_future.wait()    # Wait for Group Chain (which includes Group Pipeline)

    logger.info(f"All pipelines for {day} completed.")

if __name__ == "__main__":
    main_pipeline_flow()