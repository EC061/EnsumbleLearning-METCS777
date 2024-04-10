import sys
from pyspark import SparkContext
from pyspark.sql import SparkSession

if len(sys.argv) != 7:
    print("Usage: generateTrainAndSampleData.py <input_file_path> <output_dir> ", file=sys.stderr)
    exit(-1)

data_path = sys.argv[1]
labels_path = sys.argv[2]
outdir_1=sys.argv[3]
outdir_2=sys.argv[4]
outdir_3=sys.argv[5]
outdir_4=sys.argv[6]

sc = SparkContext(appName="generate train data")
spark = SparkSession(sc)

train_data = spark.read.csv(data_path, header=True, inferSchema=True)
train_labels = spark.read.csv(labels_path, header=True, inferSchema=True)
data = train_data.join(train_labels, on='customer_ID')

# generate train and test data with 80% and 20% split
class_0_df = data.filter(data.target == 0)
class_1_df = data.filter(data.target == 1)
class_0_train, class_0_test = class_0_df.randomSplit([0.8, 0.2], seed=2024)
class_1_train, class_1_test = class_1_df.randomSplit([0.8, 0.2], seed=2024)

train_df = class_0_train.union(class_1_train)
test_df = class_0_test.union(class_1_test)
train_df.write.mode('overwrite').parquet(outdir_1)
test_df.write.mode('overwrite').parquet(outdir_2)

# generate sample data with 10% of the original data
class_0_small, _ = class_0_df.randomSplit([0.1, 0.9], seed=2024)
class_1_small, _ = class_1_df.randomSplit([0.1, 0.9], seed=2024)
class_0_train, class_0_test = class_0_small.randomSplit([0.8, 0.2], seed=2024)
class_1_train, class_1_test = class_1_small.randomSplit([0.8, 0.2], seed=2024)
train_df_small = class_0_train.union(class_1_train)
test_df_small = class_0_test.union(class_1_test)
train_df_small.coalesce(1).write.csv(outdir_3, header=True, mode="overwrite")
test_df_small.coalesce(1).write.csv(outdir_4, header=True, mode="overwrite")