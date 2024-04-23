import sys
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier, \
                                      LinearSVC, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from pyspark.ml.pipeline import PipelineModel
import os

if len(sys.argv) != 4:
    print("Usage: train_evaluate.py <input_file_path> <output_dir> ", file=sys.stderr)
    exit(-1)

TRAIN_FILE = sys.argv[1]
TEST_FILE = sys.argv[2]
OUT_DIR1 = sys.argv[3]

sc = SparkContext(appName="Feature Engineering for Models")
spark = SparkSession(sc)
train_selected = spark.read.parquet(TRAIN_FILE)
test_selected = spark.read.parquet(TEST_FILE)

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


with open(OUT_DIR1, "w") as file:
    for name, model in models.items():
        model_path = os.path.join("models", f"{name}_model")
        if os.path.exists(model_path):
            model = PipelineModel.load(model_path)
            print(f"Loaded model: {name}")
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
            print(f"Trained and saved model: {name}")
        
        predictions = model.transform(test_selected)
        accuracy = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "accuracy"})
        precision = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "precisionByLabel"})
        recall = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "recallByLabel"})
        f1 = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "f1"})
        auc = binary_evaluator.evaluate(predictions, {binary_evaluator.metricName: "areaUnderROC"})

        result_string = f"Model: {name}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, AUC: {auc}\n"
        print(result_string)
        file.write(result_string)
        
sc.stop()