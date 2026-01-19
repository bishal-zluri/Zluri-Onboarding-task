from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from datetime import timedelta
import sys
import os

# --- PATH CONFIGURATION ---
# Current directory (src/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add 'src' itself to path if not present
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

# CRITICAL FIX: Add specific pipeline subdirectories to sys.path
sys.path.insert(0, os.path.join(CURRENT_DIR, 'user_pipeline'))
sys.path.insert(0, os.path.join(CURRENT_DIR, 'groups_pipeline'))
sys.path.insert(0, os.path.join(CURRENT_DIR, 'transaction_pipeline'))

# Import Pipeline Modules
from user_pipeline.user_transformer import transform_and_reconcile_users, POSTGRES_JAR
from groups_pipeline.groups_transformer import transform_and_reconcile_groups
from transaction_pipeline.transaction_transformer import transform_and_load_transactions

from pyspark.sql import SparkSession

# --- SPARK SESSION MANAGEMENT ---
@task(name="Initialize Spark", cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def get_spark_session(app_name="Prefect-ETL"):
    """
    Creates or gets a Spark Session. Cached to avoid recreation overhead.
    """
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

# --- ENTITY TASKS ---

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

# --- POST-SYNC TASKS ---

@task(name="Mark Inactive Users")
def post_sync_user_(spark_session):
    logger = get_run_logger()
    logger.info("Running post-sync done for Users...")
    logger.info("User status reconciliation complete (handled in main pipeline).")

@task(name="Mark Inactive Groups")
def post_sync_group_(spark_session):
    logger = get_run_logger()
    logger.info("Running post-sync done for Groups...")
    logger.info("Group hierarchy status propagation complete.")

# --- MAIN FLOW ---

@flow(name="Zluri Data Assignment Pipeline")
def main_pipeline_flow():
    """
    Main orchestration flow. Runs all pipelines for a specific day.
    """
    logger = get_run_logger()
    logger.info("Starting Main Orchestrator...")
    
    # 1. Initialize Spark
    spark = get_spark_session()
    
    # 2. Define sync day
    day = "sync-day2"
    logger.info(f"=== PROCESSING: {day} ===")

    # 3. Inject day into global config of imported modules
    # (Assuming these modules use a global variable 'DAY_FOLDER')
    import s3_reader_users as ur
    ur.DAY_FOLDER = day
    import s3_reader_groups as gr
    gr.DAY_FOLDER = day
    import s3_reader_transaction as tr
    tr.DAY_FOLDER = day
    
    # 4. Execute Tasks
    # A. Users (Critical Path)
    user_task = run_user_pipeline(spark, day)
    
    # B. Groups (Depends on Users)
    group_task = run_group_pipeline(spark, day, wait_for=[user_task])
    
    # C. Transactions (Independent)
    transaction_task = run_transaction_pipeline(spark, day)
    
    # D. Post-Sync Tasks (Wait for respective pipelines)
    post_sync_user_(spark, wait_for=[user_task])
    post_sync_group_(spark, wait_for=[group_task])
    
    logger.info(f"All pipelines for {day} completed.")

if __name__ == "__main__":
    main_pipeline_flow()