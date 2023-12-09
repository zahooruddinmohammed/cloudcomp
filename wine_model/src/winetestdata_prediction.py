import os
import sys
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.functions import col

def process_dataframe(input_dataframe):
    return input_dataframe.select(*(col(col_name).cast("double").alias(col_name.strip("\"")) for col_name in input_dataframe.columns))

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName('wine_prediction_spark_modified_app') \
        .getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    current_directory = os.getcwd()

    if len(sys.argv) > 3:
        sys.exit(-1)
    elif len(sys.argv) > 1:
        input_data_path = sys.argv[1]
        if not ("/" in input_data_path):
            input_data_path = os.path.join(current_directory, input_data_path)
        model_directory = os.path.join(current_directory, "test_model")
        print("Test data file location:")
        print(input_data_path)
    else:
        print("Current directory:")
        print(current_directory)
        input_data_path = os.path.join(current_directory, "testdata.csv")
        model_directory = os.path.join(current_directory, "test_model")

    data_frame = (spark.read
          .format("csv")
          .option('header', 'true')
          .option("sep", ";")
          .option("inferschema", 'true')
          .load(input_data_path))

    processed_df = process_dataframe(data_frame)

    selected_features_list = [
        'fixed_acidity',
        'volatile_acidity',
        'citric_acid',
        'chlorides',
        'total_sulfur_dioxide',
        'density',
        'sulphates',
        'alcohol',
    ]

    trained_ml_model = PipelineModel.load(model_directory)

    predictions_result = trained_ml_model.transform(processed_df)
    print("Sample predictions:")
    print(predictions_result.show(5))

    results_df = predictions_result.select(['prediction', 'label'])
    evaluator_metric = MulticlassClassificationEvaluator(
        labelCol='label',
        predictionCol='prediction',
        metricName='accuracy'
    )
    accuracy_result = evaluator_metric.evaluate(predictions_result)
    print('Test Accuracy of the wine prediction model:', accuracy_result)

    metrics_result = MulticlassMetrics(results_df.rdd.map(tuple))
    print('Weighted F1 score of the wine prediction model:', metrics_result.weightedFMeasure())
    sys.exit(0)
