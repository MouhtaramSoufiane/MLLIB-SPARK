package org.example;

import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class Main {
    public static void main(String[] args) {
        SparkSession session= SparkSession.builder()
                .appName("tp spark ml")
                .master("local[*]")
                .getOrCreate();

        Dataset<Row> dataset=session.read().option("inferSchema", true).option("header",true).csv("advertising.csv");

        VectorAssembler vectorAssembler=new VectorAssembler().setInputCols(
                new String[]{"TV","Radio","Newspaper"}
        ).setOutputCol("Features");

        Dataset<Row> assembledDS = vectorAssembler.transform(dataset);
        Dataset<Row> splits[] = assembledDS.randomSplit(new double[]{0.8, 0.2}, 123);

        Dataset<Row> train = splits[0];
        Dataset<Row> test = splits[1];

        LinearRegression regression=new LinearRegression().setLabelCol("Sales").setFeaturesCol("Features");
        LinearRegressionModel model = regression.fit(train);
        Dataset<Row> predictions = model.transform(test);
        predictions.show();
        System.out.println("Intercept="+model.intercept()+" coefficients="+model.coefficients());


    }
}