import sys
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import Bucketizer

def main(test_data_path, planes_path, airports_path):
    # ---------------------------------------------------------
    # 1. Initialize Spark Session
    # ---------------------------------------------------------
    spark = SparkSession.builder\
        .appName("FlightDelayPerformanceTest")\
        .config("spark.ui.showConsoleProgress", "false")\
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    print(f"\n[INFO] Starting Inference Engine...")
    print(f"[INFO] Loading Test Data from: {test_data_path}")

    # ---------------------------------------------------------
    # 2. Robust Data Loading (Explicit Schema)
    # ---------------------------------------------------------
    flight_schema = StructType([
        StructField("Year", IntegerType(), True),
        StructField("Month", IntegerType(), True),
        StructField("DayofMonth", IntegerType(), True),
        StructField("DayOfWeek", IntegerType(), True),
        StructField("DepTime", DoubleType(), True),
        StructField("CRSDepTime", DoubleType(), True),
        StructField("ArrTime", DoubleType(), True),
        StructField("CRSArrTime", DoubleType(), True),
        StructField("UniqueCarrier", StringType(), True),
        StructField("FlightNum", IntegerType(), True),
        StructField("TailNum", StringType(), True),
        StructField("ActualElapsedTime", DoubleType(), True),
        StructField("CRSElapsedTime", DoubleType(), True),
        StructField("AirTime", DoubleType(), True),
        StructField("ArrDelay", DoubleType(), True),  # Target
        StructField("DepDelay", DoubleType(), True),
        StructField("Origin", StringType(), True),
        StructField("Dest", StringType(), True),
        StructField("Distance", DoubleType(), True),
        StructField("TaxiIn", DoubleType(), True),
        StructField("TaxiOut", DoubleType(), True),
        StructField("Cancelled", DoubleType(), True),
        StructField("CancellationCode", StringType(), True),
        StructField("Diverted", DoubleType(), True),
        StructField("CarrierDelay", DoubleType(), True),
        StructField("WeatherDelay", DoubleType(), True),
        StructField("NASDelay", DoubleType(), True),
        StructField("SecurityDelay", DoubleType(), True),
        StructField("LateAircraftDelay", DoubleType(), True)
    ])

    try:
        test_raw = spark.read.csv(test_data_path, header=True, schema=flight_schema)
        # Use header=True for planes/airports to ensure column names are correct
        planes_raw = spark.read.csv(planes_path, header=True, inferSchema=True)
        airports_raw = spark.read.csv(airports_path, header=True, inferSchema=True)
    except Exception as e:
        print(f"[ERROR] Failed to load datasets: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # 3. ETL & Feature Engineering
    # ---------------------------------------------------------
    print("[INFO] Processing Data (Cleaning & Feature Engineering)...")
    
    # Clean: Remove Cancelled/Diverted and rows with missing critical info
    df_clean = test_raw.filter("Cancelled == 0 AND Diverted == 0") \
                       .dropna(subset=["ArrDelay", "DepDelay", "Distance", "DepTime"])
    
    # 3a. Basic Time & Weather Features (PRE-JOIN)
    # Note: We cannot calculate IceRisk yet because we don't have Latitudes
    df_feat = df_clean.withColumn("DepHour", (F.col("DepTime") / 100).cast("int")) \
                      .withColumn("IsWinter", F.when(F.col("Month").isin(12, 1, 2), 1).otherwise(0)) \
                      .withColumn("IsSummer", F.when(F.col("Month").isin(6, 7, 8), 1).otherwise(0))

    # 3b. Join Plane Metadata (Asset Info)
    planes_sel = planes_raw.select(
        F.col("tailnum"), 
        F.col("year").alias("year_manufactured"), 
        F.col("manufacturer")
    )
    df_feat = df_feat.join(planes_sel, df_feat.TailNum == planes_sel.tailnum, how="left")

    # 3c. Join Airport Metadata (Geo Info)
    # We need Latitudes to calculate IceRisk and State for the model
    airports_meta = airports_raw.select(
        F.col("iata").alias("Code"), 
        F.col("lat").alias("Lat"), 
        F.col("state").alias("State")
    ).distinct()

    # Join Origin (Get OriginLat and OriginState)
    df_feat = df_feat.join(airports_meta, df_feat.Origin == airports_meta.Code, how="left") \
                     .withColumnRenamed("Lat", "OriginLat") \
                     .withColumnRenamed("State", "OriginState") \
                     .drop("Code")
    
    # Join Destination (Get DestLat and DestState)
    dest_meta = airports_meta.select(
        F.col("Code").alias("DestCode"), 
        F.col("Lat").alias("DestLat"), 
        F.col("State").alias("DestState")
    )
    df_feat = df_feat.join(dest_meta, df_feat.Dest == dest_meta.DestCode, how="left").drop("DestCode")

    # 3d. Complex Features (Now that we have all columns)

    # Ice Risk: Winter + High Latitude (> 40 degrees)
    df_feat = df_feat.withColumn("TripMaxLat", F.greatest(F.col("OriginLat"), F.col("DestLat")))
    df_feat = df_feat.withColumn("IceRisk", 
        F.when((F.col("IsWinter") == 1) & (F.col("TripMaxLat") >= 40), 1).otherwise(0)
    )

    # Hub Status (Hardcoded Top 20 from Training)
    hub_list = ['ATL', 'ORD', 'DFW', 'DEN', 'LAX', 'PHX', 'IAH', 'LAS', 'DTW', 'SLC', 'SFO', 'EWR', 'MCO', 'MSP', 'CLT', 'LGA', 'BOS', 'JFK', 'BWI', 'SEA']
    df_feat = df_feat.withColumn("Origin_IsHub", F.when(F.col("Origin").isin(hub_list), 1).otherwise(0))
    df_feat = df_feat.withColumn("Dest_IsHub", F.when(F.col("Dest").isin(hub_list), 1).otherwise(0))

    # Plane Age & Cleaning Manufacturer
    df_feat = df_feat.withColumn("PlaneAge", F.col("year").cast("int") - F.col("year_manufactured"))
    median_age = df_feat.approxQuantile("PlaneAge", [0.5], 0.01)[0]
    df_feat = df_feat.fillna(median_age, subset=["PlaneAge"])

    df_feat = df_feat.withColumn("Manufacturer_Clean", 
        F.when(F.col("manufacturer").like("%AIRBUS%"), "AIRBUS")
        .when(F.col("manufacturer").like("%BOEING%"), "BOEING")
        .when(F.col("manufacturer").like("%BOMBARDIER%"), "BOMBARDIER")
        .when(F.col("manufacturer").like("%EMBRAER%"), "EMBRAER")
        .when(F.col("manufacturer").like("%MCDONNELL%"), "MCDONNELL_DOUGLAS")
        .otherwise("OTHER")
    )

    # 3e. Bucketizer for Plane Age
    bucketizer = Bucketizer(splits=[-float("inf"), 23, 33, float("inf")], inputCol="PlaneAge", outputCol="AgeBin")
    df_feat = bucketizer.transform(df_feat)
    df_final = df_feat.fillna({'AgeBin': -1})

    
    # Fill N/A for features created by joins
    df_final = df_final.fillna(0, subset=["Origin_IsHub", "Dest_IsHub", "IceRisk", "IsWinter", "IsSummer"])

    # ---------------------------------------------------------
    # 4. Model Inference
    # ---------------------------------------------------------
    print("[INFO] Loading Best Model Pipeline...")
    try:
        model = PipelineModel.load("models/best_model") # Ensure this matches your folder name!
    except Exception as e:
        print(f"[ERROR] Model not found. Error: {e}")
        sys.exit(1)

    print("[INFO] Generating Predictions...")
    try:
        predictions = model.transform(df_final)
    except Exception as e:
        print(f"[ERROR] Schema Mismatch! The model expected columns that are missing.")
        print(f"Error details: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # 5. Output & Evaluation
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(f"{'PREDICTION SAMPLE (First 20)':^50}")
    print("="*50)
    
    predictions.select(
        F.col("Origin"), 
        F.col("Dest"), 
        F.col("ArrDelay").alias("Real_Delay"), 
        F.col("prediction").alias("Model_Pred")
    ).show(20)

    # Metric Calculation
    evaluator_rmse = RegressionEvaluator(labelCol="ArrDelay", predictionCol="prediction", metricName="rmse")
    evaluator_r2 = RegressionEvaluator(labelCol="ArrDelay", predictionCol="prediction", metricName="r2")
    
    rmse = evaluator_rmse.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)

    print("="*50)
    print(f"{'FINAL PERFORMANCE AUDIT':^50}")
    print("="*50)
    print(f" RMSE : {rmse:.4f} minutes")
    print(f" R²   : {r2:.4f}")
    print("="*50 + "\n")

    spark.stop()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: spark-submit app.py <test_data_csv> [planes_csv] [airports_csv]")
        sys.exit(1)
    
    test_path = sys.argv[1]
    p_path = sys.argv[2] if len(sys.argv) > 2 else "../../training_data/documentation/plane-data.csv"
    a_path = sys.argv[3] if len(sys.argv) > 3 else "../../training_data/documentation/airports.csv"

    if len(sys.argv) < 3:
        print(f"\n[NOTE] Using default auxiliary paths:")
        print(f" -> Planes:   {p_path}")
        print(f" -> Airports: {a_path}\n")
    
    main(test_path, p_path, a_path)