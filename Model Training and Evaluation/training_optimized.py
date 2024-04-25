from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import mean, stddev, min, max, last, count, countDistinct, col, lit, struct, avg
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier, \
                                      LinearSVC, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from pyspark.ml.pipeline import PipelineModel
from functools import reduce
import os
CAT_VARS = ["B_30","B_38","D_114","D_116","D_117","D_120","D_126","D_63","D_64","D_66","D_68"]
TARGET_COLUMN = 'target'
TRAIN_FILE = 'train_small.txt'
TEST_FILE = 'test_small.txt'
CAT_FEATURES = ['B_30_last', 'B_38_last', 'D_114_last', 'D_116_last', 'D_117_last',
                'D_120_last', 'D_126_last', 'D_63_last', 'D_64_last','D_66_last', 'D_68_last']
conf = SparkConf() \
    .setAppName("Train and Save Models") \
    .set("spark.executor.memory", "8g") \
    .set("spark.driver.memory", "8g")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)
sc.setLogLevel("ERROR")
if len(sys.argv) != 5:
    print("Usage: train.py <input_file_path> <input_file_path2> <output_dir> <output_dir2> ", file=sys.stderr)
    exit(-1)

TRAIN_FILE = sys.argv[1]
TEST_FILE = sys.argv[2]
OUT_DIR1 = sys.argv[3]
MODEL_DIR = sys.argv[4]
data_train = spark.read.csv(TRAIN_FILE, header=True, inferSchema=True)
data_test = spark.read.csv(TEST_FILE, header=True, inferSchema=True)
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
indexers = [
    StringIndexer(inputCol=column, outputCol=column + "_indexed", handleInvalid='keep')
    for column in CAT_FEATURES
]
continuous_features = [c for c in data_train.columns if c not in CAT_FEATURES and c != TARGET_COLUMN and not c.endswith('_indexed')]
data_train = data_train.fillna(0)
data_test = data_test.fillna(0)
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
feature_vector_size = len(train_selected.select("features").first()[0])
layers = [feature_vector_size, feature_vector_size // 2 + 1, feature_vector_size // 4 + 1, 2]


models = {
    "GBTClassifier": GBTClassifier(featuresCol='features', labelCol='target'),
    "LinearSVC": LinearSVC(featuresCol='features', labelCol='target'),
    "LogisticRegression": LogisticRegression(featuresCol='features', labelCol='target'),
    "DecisionTreeClassifier": DecisionTreeClassifier(featuresCol='features', labelCol='target'),
    "RandomForestClassifier": RandomForestClassifier(featuresCol='features', labelCol='target'),
    "MultilayerPerceptronClassifier": MultilayerPerceptronClassifier(featuresCol='features', labelCol='target', layers=layers)
}

paramGrids = {
    "GBTClassifier": ParamGridBuilder() \
        .addGrid(GBTClassifier.maxDepth, [2, 5, 10]) \
        .addGrid(GBTClassifier.maxBins, [10, 20, 40]) \
        .build(),
        
    "LinearSVC": ParamGridBuilder() \
        .addGrid(LinearSVC.maxIter, [10, 100, 1000]) \
        .addGrid(LinearSVC.regParam, [0.1, 0.01]) \
        .build(),
        
    "LogisticRegression": ParamGridBuilder() \
        .addGrid(LogisticRegression.maxIter, [10, 100, 1000]) \
        .addGrid(LogisticRegression.regParam, [0.1, 0.01]) \
        .addGrid(LogisticRegression.elasticNetParam, [0.0, 0.5, 1.0]) \
        .build(),
        
    "DecisionTreeClassifier": ParamGridBuilder() \
        .addGrid(DecisionTreeClassifier.maxDepth, [2, 5, 10]) \
        .addGrid(DecisionTreeClassifier.maxBins, [10, 20, 40]) \
        .build(),
        
    "RandomForestClassifier": ParamGridBuilder() \
        .addGrid(RandomForestClassifier.numTrees, [10, 50, 100]) \
        .addGrid(RandomForestClassifier.maxDepth, [2, 5, 10]) \
        .addGrid(RandomForestClassifier.maxBins, [10, 20, 40]) \
        .build(),
        
    "MultilayerPerceptronClassifier": ParamGridBuilder() \
        .addGrid(MultilayerPerceptronClassifier.maxIter, [100, 200, 300]) \
        .addGrid(MultilayerPerceptronClassifier.blockSize, [128, 256]) \
        .build()
}

binary_evaluator = BinaryClassificationEvaluator(labelCol="target")
multi_evaluator = MulticlassClassificationEvaluator(labelCol="target")

results = []
for name, model in models.items():
    model_path = os.path.join("models", f"{name}_model")
    if os.path.exists(model_path):
        print(f"Loading model: {model_path}")
        model = PipelineModel.load(model_path)
        print(f"Loaded model: {name}")
    else:
        pipeline = Pipeline(stages=[model])
        paramGrid = paramGrids[name]
        crossval = CrossValidator(estimator=pipeline,
                                    estimatorParamMaps=paramGrid,
                                    evaluator=multi_evaluator,
                                    numFolds=3, parallelism=4)
        print("hi")
        cvModel = crossval.fit(train_selected)
        model = cvModel.bestModel
        model.save(model_path)
        print(f"Trained and saved model: {name}")

    predictions = model.transform(test_selected)
    accuracy = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "accuracy"})
    precision = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "precisionByLabel"})
    recall = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "recallByLabel"})
    f1 = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "f1"})
    auc = binary_evaluator.evaluate(predictions, {binary_evaluator.metricName: "areaUnderROC"})

    best_params = model.stages[-1].extractParamMap()
    param_str = ", ".join([f"{p.name}: {best_params[p]}" for p in best_params])
    result_string = (f"Model: {name}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, "
                        f"F1 Score: {f1}, AUC: {auc}\n")
    parameters_string = f"Best Parameters: {param_str}\n"
    results.append(Row(model=name, accuracy=accuracy, precision=precision, recall=recall, f1=f1, auc=auc))
    results.append(Row(model=name, parameters=parameters_string))

results_df = spark.createDataFrame(results)
results_df.write.text(OUT_DIR1)



model_names = ["GBTClassifier", "LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier", "MultilayerPerceptronClassifier"]
models = {name: PipelineModel.load(os.path.join("models", f"{name}_model")) for name in model_names}

predictions = [model.transform(test_selected).withColumnRenamed('probability', f'probability_{name}') for name, model in models.items()]
combined_predictions = reduce(lambda df1, df2: df1.join(df2.drop("prediction"), "id"), predictions)

num_models = len(models)
average_probability = combined_predictions.select(avg(struct([col(f"probability_{name}") for name in model_names]))).alias("avg_probability")

final_prediction = average_probability.withColumn('final_prediction', (col('avg_probability') > lit(0.5)).cast("integer"))

accuracy = multi_evaluator.evaluate(final_prediction, {multi_evaluator.metricName: "accuracy"})
precision = multi_evaluator.evaluate(final_prediction, {multi_evaluator.metricName: "precisionByLabel"})
recall = multi_evaluator.evaluate(final_prediction, {multi_evaluator.metricName: "recallByLabel"})
f1 = multi_evaluator.evaluate(final_prediction, {multi_evaluator.metricName: "f1"})
auc = binary_evaluator.evaluate(final_prediction, {binary_evaluator.metricName: "areaUnderROC"})

result_string = f"Ensemble Model - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, AUC: {auc}"
print(result_string)

result_string = spark.createDataFrame(result_string)
result_string.write.text(OUT_DIR1) 
sc.stop()
