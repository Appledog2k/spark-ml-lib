package appledog.research.spark.application;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class SparkML {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("JavaSparkML")
                .master("local[*]")
                .getOrCreate();

        // Create dataframe from csv file
        String path = "src/main/resources/diabetes.csv";

        // Create df header = true, inferschema = true
        Dataset<Row> data = spark.read().option("header", "true").option("inferSchema", "true").csv(path);
        data.cache();

        // Cleaning data

        System.out.println("CLEANING DATA \n");

        // find null values
        System.out.println("Finding null values");
        for (String col : data.columns()) {
            // find null values
            long nullCount = data.filter(data.col(col).isNull()).count();

            // print name column and null count
            System.out.println(col + " has " + nullCount + " null values");
        }
        System.out.println();

        // find total number of 0 values in columns: Glucose, BloodPressure, SkinThickness, Insulin, BMI
        System.out.println("Finding zero values");
        for (String col : new String[]{"Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"}) {
            long zeroCount = data.filter(data.col(col).equalTo(0)).count();
            System.out.println(col + " has " + zeroCount + " zero values");
        }
        System.out.println();

        // caculate mean of columns: Glucose, BloodPressure, SkinThickness, Insulin, BMI format integer
        System.out.println("Calculating mean of columns");
        for (String col : new String[]{"Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"}) {
            double mean = data.agg(org.apache.spark.sql.functions.mean(col)).first().getDouble(0);
            System.out.println(col + " has mean value: " + (int) mean);
            // if value is 0, replace with mean
            data = data.withColumn(col, org.apache.spark.sql.functions.when(data.col(col).equalTo(0), (int) mean).otherwise(data.col(col)));
        }

        data.show();
    }
}
