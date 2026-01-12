# DE Intern Onboarding - Jan 2026

# Problem Statement

Build a data pipeline that ingests integration data, transforms it into Zluri entities, and stores them in a database.

At Zluri, we fetch data from dozens of integrations (Okta, GSuite, Azure AD, etc.) and process it into standardized entities like Users, Groups, and Transactions. Using Python, PySpark, and Prefect, build a pipeline system that can ingest raw data from S3, transform it appropriately, and store the processed entities in a database of your choice.

**Before You Start:**

1. **Explore Zluri UI** — You’ll get credentials to explore the actual product. Look at how Users, Groups, and Transactions appear. Design your own simplified schema based on what you see (not all fields required).
    - URL: app.zluri.dev
    - Creds: 
    [williamballard@zluri.dev](mailto:williamballard@zluri.dev)
    9BRQagwr0WRT4geBbOAH3A
2. **Choose Your Database** — Postgres or MongoDB (or both). Document your choice with pros/cons. There’s no right answer—we want to see your reasoning.
3. **Access Mock Data** — S3 path will be shared. Use AWS CLI to download the data locally.
    
    S3 Path: `s3://zluri-data-assignment/assignment-jan-2026/`
    

Please note: Feel free to experiment with the schema design. The Zluri UI exploration is to help you visualize the entities.

## Functional Requirements

### 1. User Pipeline

**As a pipeline, it should be able to:**

- Ingest user data from the source files (Day 1 and Day 2).
- Parse and validate the schema of incoming records.
- Transform raw data into the target User entity schema.
- Load the transformed data into the database.
- Handle duplicate users gracefully (upsert based on unique identifier).
- Compare Day 1 and Day 2 syncs — if a user exists in Day 1 but is missing in Day 2, mark as `inactive`.

**Key Requirements:**

- State management across syncs for inactive marking.
- Schema validation with clear error logging for malformed records.
- Idempotent pipeline — re-running should not create duplicates.

---

### 2. Group Pipeline

**As a pipeline, it should be able to:**

- Ingest group data from the source files (Day 1 and Day 2).
- Parse and validate the schema of incoming records.
- Handle hierarchical data (groups can have parent groups at arbitrary depth).
- Process group membership data (which users belong to which groups).
- Transform raw data into the target Group entity schema.
- Load the transformed data into the database (including membership relationships).
- Mark groups as `inactive` based on the following logic:
    - A group is inactive if it has **no active users** as direct members
    - **AND** none of its child groups (at any depth) have active users

**Key Requirements:**

- Support for nested group hierarchy (Bonus: arbitrary depth using recursive queries).
- Many-to-many relationship handling for User-Group membership.
- Hierarchical inactive logic — must traverse child groups to determine parent group status.

---

### 3. Transaction Pipeline

**As a pipeline, it should be able to:**

- Ingest transaction data from the source files (Day 1 and Day 2).
- Process only **delta/incremental** records (not full reload).
- Parse and validate the schema of incoming records.
- **Convert all transaction amounts to USD** — transactions may come in different currencies (EUR, GBP, INR, etc.).
- Fetch exchange rates for the transaction date and apply conversion.
- Store both original amount (with currency) and converted USD amount.
- Deduplicate transactions — no two transactions with the same unique key should exist.
- Transform raw data into the target Transaction entity schema.
- Load only new/updated transactions into the database.
- Track what has been processed to avoid reprocessing in future runs.

**Key Requirements:**

- **Currency conversion** — fetch exchange rates from a public API (e.g., exchangerate-api.com, open.er-api.com) or use a static rates file.
- **Note**: Exchange rates change daily. Use the rate for the transaction date, not the processing date.
- Store both `original_amount`, `original_currency` and `amount_usd` for audit purposes.
- Incremental load logic (watermark-based or unique key-based deduplication).
- Idempotent processing — re-running should not create duplicates.
- Clear logging of records processed, skipped, and failed.

---

### 4. Pipeline Orchestration (Prefect)

**As an orchestration layer, it should be able to:**

- Define a Prefect flow that runs all three entity pipelines.
- Respect dependencies between pipelines (e.g., Groups depend on Users for membership).
- Handle Day 1 → Day 2 sync simulation in sequence.
- Run post-sync tasks to mark inactive entities.
- Provide error handling and retries for failed tasks.
- Log pipeline execution status and metrics.

**Key Requirements:**

- Proper task dependencies (DAG structure).
- Error handling with meaningful error messages.
- Observability — ability to track what ran, what failed, what succeeded.

**Pipeline Dependency Order:**

```
Day N Sync
    │
    ├── Users (no dependencies)
    │
    ├── Groups (after Users — needs user IDs for membership)
    │
    └── Transactions (independent, can run parallel with Groups/Users)

Post-Sync
    │
    ├── Mark inactive Users (compare with previous sync)
    │
    └── Mark inactive Groups (check for active users + child group hierarchy)
```

---

### 5. Data Quality & Validation

**As a data engineer, you should ensure:**

- Schema validation on all incoming records.
- Referential integrity (Group Memberships → Users).
- No duplicate records in final tables.
- Clear logging and error reporting for bad records.
- Unit tests covering transformation logic with at least 80% coverage.

**Key Requirements:**

- Validation should not silently drop records — log all issues.
- Bad records should be quarantined or reported, not mixed with good data.
- Tests should cover edge cases (nulls, missing fields, invalid references).

---

# Milestones

## Week 1 - Deliverables

- Setup the development environment (Python, PySpark, Prefect, Database)
- Setup the project repository with proper structure
- Create schema design document based on Zluri UI exploration
- Implement User pipeline (ingest → transform → load)
- Implement inactive User logic (Day 1 vs Day 2 comparison)
- Implement Group pipeline (hierarchy + user membership)
- Unit testing (Use AI to help you do this)
    - Setup pytest framework for unit testing
    - Write tests for transformation logic
- End of Week 1 demo to showcase User & Group pipelines, query DB to prove data landed

## Week 2 - Deliverables

- Implement inactive logic for Groups (hierarchical check for active users)
- Implement Transaction pipeline (incremental processing + deduplication + currency conversion)
- Complete Prefect orchestration (full DAG with all pipelines)
- Unit testing (Use AI to help you do this)
    - Ensure at least 80% branch coverage for unit testing
- Documentation
    - Database choice rationale with pros/cons
    - Schema design decisions
    - Pipeline architecture overview
- End of Week 2 demo to showcase fully functional pipeline system

## End of Week 2 → Bug Bash

- Test and try to break each other’s pipelines to earn the most amount of points
- Feed edge-case data (malformed records, circular hierarchies, duplicates)
- Look for logic bugs (incorrect inactive marking, broken memberships, missed deduplication)

## **⚠️ Performance Testing:**

After the Week 2 demo, you will be given a **large dataset** via S3. Your pipeline will be evaluated on:

- Total execution time
- Spark job efficiency (partitioning, shuffle, caching)
- Memory usage
- Pipeline reliability at scale

Keep this in mind while designing your pipelines — think about partitioning strategies, efficient joins, and avoiding unnecessary shuffles.

## Evaluation Criteria

### Quality (200 points)

- Unit testing: Branch Coverage (100 Points)
    - >90% branch coverage will result in 100 points, 50% branch coverage will result in 37.5 points
- Number of bugs found in the bug bash (50 Points):
    - Highest person will earn 50 points, relative point allocation post that
- Performance (50 Points):
    - Best overall performance will earn 50 points, relative point allocation post that

### Code Hygiene (50 points total)

- Following the best practices while coding such as
    - Good variable and function naming conventions
    - Defining functions with single responsibility principles
    - Ensuring the code is easy to follow
    - Modular code structure (separate transformation logic from orchestration)
- Feedback will be given by your buddy

### Design Decisions (50 points)

- Database choice rationale (why Postgres/MongoDB/both)
- Schema design decisions and trade-offs
- Pipeline architecture decisions

# Bonus Challenges

Completed the core requirements? Try these for extra learning:

- **Arbitrary depth hierarchy** — handle unlimited nesting for Groups using recursive queries (Postgres) or tree traversal (MongoDB)
- **Prefect UI deployment** — run Prefect server locally, view your flows in the dashboard
- **Advanced data quality checks** — schema validation framework, referential integrity checks with detailed reporting
- **Pipeline metrics** — track and report records processed, failed, skipped per run
- **Spark optimization** — implement caching, broadcast joins, optimal partitioning for large datasets

# Prizes

Goodies will be given to the Top 1 intern 

# Tech Stack

| Tool | Purpose | Install |
| --- | --- | --- |
| Python 3.10+ | Primary language | - |
| PySpark | Data transformation | `pip install pyspark` |
| Prefect 3.x | Orchestration | `pip install prefect` |
| Postgres | Relational DB (option) | Docker or local |
| MongoDB | Document DB (option) | Docker or local |
| pytest | Unit testing | `pip install pytest pytest-cov` |

# Useful Learning Links

## Common

### Python

1. [Python Basics - W3Schools](https://www.w3schools.com/python/)
2. [Real Python Tutorials](https://realpython.com/)

### Git

[Git & GitHub Crash Course](https://youtu.be/RGOj5yH7evk?feature=shared)

### JSON

[JSON Tutorial](https://beginnersbook.com/2015/04/json-tutorial/)

## Data Engineering

### PySpark

1. [Official DataFrame Quickstart](https://spark.apache.org/docs/latest/api/python/getting_started/quickstart_df.html) — Start here
2. [Real Python: First Steps with PySpark](https://realpython.com/pyspark-intro/)
3. [Databricks PySpark Basics](https://docs.databricks.com/en/pyspark/basics.html)

### Prefect

1. [Official Quickstart](https://docs.prefect.io/v3/get-started/quickstart) — Start here
2. [Prefect Concepts: Flows & Tasks](https://docs.prefect.io/v3/develop/write-flows)
3. [DataCamp: Workflow Orchestration with Prefect](https://www.datacamp.com/tutorial/ml-workflow-orchestration-with-prefect)

### PostgreSQL

1. [Official Tutorial](https://www.postgresql.org/docs/current/tutorial.html)
2. [PostgreSQL Recursive Queries (for hierarchy)](https://www.postgresql.org/docs/current/queries-with.html)
3. [PostgreSQL Tutorial Site](https://www.postgresqltutorial.com/)

### MongoDB

1. [MongoDB University](https://learn.mongodb.com/) — Free courses, highly recommended
2. [Embed vs Reference Guide](https://www.mongodb.com/docs/manual/data-modeling/concepts/embedding-vs-references/)
3. [Real Python: MongoDB and Python](https://realpython.com/introduction-to-mongodb-and-python/)

### Data Engineering Concepts

1. [ETL vs ELT Explained](https://www.integrate.io/blog/etl-vs-elt/)
2. [Idempotent Data Pipelines - Start Data Engineering](https://www.startdataengineering.com/post/why-how-idempotent-data-pipeline/)
3. [Idempotency in Data Pipelines - Prefect](https://www.prefect.io/blog/the-importance-of-idempotent-data-pipelines-for-resilience)

### Unit Testing (pytest)

[pytest Documentation](https://docs.pytest.org/en/stable/getting-started.html)

---

| **Intern Name** | **Buddy** |
| --- | --- |
| Ashutosh | Kunj |
| Bishal | Deepanshu |
| Rithvik | Abhineet |