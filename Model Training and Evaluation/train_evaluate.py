import sys
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier, \
                                      LinearSVC, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql import Row
import os

if len(sys.argv) != 5:
    print("Usage: train_evaluate.py <input_file_path> <output_dir> ", file=sys.stderr)
    exit(-1)

TRAIN_FILE = sys.argv[1]
TEST_FILE = sys.argv[2]
OUT_DIR1 = sys.argv[3]
MODEL_DIR = sys.argv[4]

sc = SparkContext(appName="Feature Engineering for Models")
spark = SparkSession(sc)
train_selected = spark.read.parquet(TRAIN_FILE).cache()
test_selected = spark.read.parquet(TEST_FILE).cache()

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
    model_path = os.path.join(MODEL_DIR, f"{name}_model")
    if os.path.exists(model_path):
        model = PipelineModel.load(model_path)
    else:
        pipeline = Pipeline(stages=[model])
        paramGrid = paramGrids[name]
        crossval = CrossValidator(estimator=pipeline,
                                    estimatorParamMaps=paramGrid,
                                    evaluator=multi_evaluator,
                                    numFolds=3)
        cvModel = crossval.fit(train_selected)
        model = cvModel.bestModel
        model.save(model_path)
    
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
sc.stop()