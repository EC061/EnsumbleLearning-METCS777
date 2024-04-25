import os
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark import SparkContext
import sys

if len(sys.argv) != 5:
    print("Usage: ensemble_train.py <input_file_path> <output_dir> ")
    exit(-1)

TRAIN_FILE = sys.argv[1]
TEST_FILE = sys.argv[2]
OUT_DIR1 = sys.argv[3]
MODEL_DIR = sys.argv[4]

sc = SparkContext(appName="Ensemble Training for Models")
spark = SparkSession(sc)
test_selected = spark.read.parquet(TEST_FILE).cache()

models = ["GBTClassifier", "LinearSVC", "LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier"]

combined_df = None
for i, model_name in enumerate(models):
    model_path = os.path.join(MODEL_DIR, f"{model_name}_model")
    print(f"Loading model: {model_name}")
    model = PipelineModel.load(model_path)
    prediction_column = f"prediction_{i}"
    predictions = model.transform(test_selected).withColumnRenamed("prediction", prediction_column)
    if combined_df is None:
        combined_df = predictions.select("target", prediction_column)
    else:
        combined_df = combined_df.join(predictions.select("target", prediction_column), "target")

prediction_columns = [f"prediction_{i}" for i in range(len(models))]
combined_df = combined_df.withColumn("ensemble_prediction", F.expr('+'.join(prediction_columns))/len(prediction_columns))

combined_df = combined_df.withColumn("final_prediction", (F.col("ensemble_prediction") > 0.5).cast("double"))
predictions = combined_df.select("target", "final_prediction").withColumnRenamed("final_prediction", "prediction").cache()

multi_evaluator = MulticlassClassificationEvaluator(labelCol="target")

accuracy = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "accuracy"})
precision = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "precisionByLabel"})
recall = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "recallByLabel"})
f1 = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "f1"})

name = "Ensemble Classifier"
results = [f"Model: {name}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}"]
results_rdd = sc.parallelize(results)
results_rdd.coalesce(1).saveAsTextFile(OUT_DIR1)
sc.stop()