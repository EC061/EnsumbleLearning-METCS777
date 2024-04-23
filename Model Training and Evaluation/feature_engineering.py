from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import mean, stddev, min, max, last, count, countDistinct
from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler
from pyspark.ml import Pipeline
import sys

if len(sys.argv) != 5:
    print("Usage: feature_engineering.py <input_file_path> <output_dir> ", file=sys.stderr)
    exit(-1)

CAT_VARS = ["B_30","B_38","D_114","D_116","D_117","D_120","D_126","D_63","D_64","D_66","D_68"]
TARGET_COLUMN = 'target'
CAT_FEATURES = ['B_30_last', 'B_38_last', 'D_114_last', 'D_116_last', 'D_117_last',
                'D_120_last', 'D_126_last', 'D_63_last', 'D_64_last','D_66_last', 'D_68_last']
TRAIN_FILE = sys.argv[1]
TEST_FILE = sys.argv[2]
OUT_DIR1 = sys.argv[3]
OUT_DIR2 = sys.argv[4]

sc = SparkContext(appName="Feature Engineering for Models")
spark = SparkSession(sc)
data_train = spark.read.parquet(TRAIN_FILE, header=True, inferSchema=True)
data_test = spark.read.parquet(TEST_FILE, header=True, inferSchema=True)
data_train = data_train.fillna(0)
data_test = data_test.fillna(0)

def feature_engineer_spark(df, CAT_VARS, TARGET_COLUMN):
    all_cols = [c for c in df.columns if c not in ['customer_ID', 'S_2']]
    cont_vars = [c for c in all_cols if c not in CAT_VARS + [TARGET_COLUMN]]
    cont_vars_agg_exprs = [expr for c in cont_vars for expr in (
        mean(c).alias(c + '_mean'),
        stddev(c).alias(c + '_std'),
        min(c).alias(c + '_min'),
        max(c).alias(c + '_max'),
        last(c).alias(c + '_last')
    )]
    cont_vars_agg = df.groupBy("customer_ID").agg(*cont_vars_agg_exprs)
    cat_vars_agg_exprs = [expr for c in CAT_VARS for expr in (
        count(c).alias(c + '_count'),
        last(c).alias(c + '_last'),
        countDistinct(c).alias(c + '_nunique')
    )]
    cat_vars_agg = df.groupBy("customer_ID").agg(*cat_vars_agg_exprs)
    df_agg = cont_vars_agg.join(cat_vars_agg, "customer_ID")
    target_column_df = df.select("customer_ID", TARGET_COLUMN)
    df_agg = df_agg.join(target_column_df, "customer_ID")
    return df_agg

data_train = feature_engineer_spark(data_train, CAT_VARS, TARGET_COLUMN).drop('customer_ID')
data_test = feature_engineer_spark(data_test, CAT_VARS, TARGET_COLUMN).drop('customer_ID')
data_train = data_train.fillna(0)
data_test = data_test.fillna(0)

indexers = [
    StringIndexer(inputCol=column, outputCol=column + "_indexed", handleInvalid='keep')
    for column in CAT_FEATURES
]
continuous_features = [c for c in data_train.columns if c not in CAT_FEATURES and c != TARGET_COLUMN and not c.endswith('_indexed')]
assembler_cont = VectorAssembler(inputCols=continuous_features, outputCol="features_raw")
scaler = MinMaxScaler(inputCol="features_raw", outputCol="scaled_features", min=0.1, max=0.9)
final_feature_columns = [col + "_indexed" for col in CAT_FEATURES] + ["scaled_features"]
assembler_final = VectorAssembler(inputCols=final_feature_columns, outputCol="features")
pipeline = Pipeline(stages=indexers + [assembler_cont, scaler, assembler_final])
model = pipeline.fit(data_train)
train_indexed = model.transform(data_train)
test_indexed = model.transform(data_test)
train_selected = train_indexed.select("features", TARGET_COLUMN).cache()
test_selected = test_indexed.select("features", TARGET_COLUMN).cache()

train_selected.write.parquet(OUT_DIR1, mode="overwrite")
test_selected.write.parquet(OUT_DIR2, mode="overwrite")
sc.stop()