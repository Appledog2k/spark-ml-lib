package appledog.research.spark.application;

import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.feature.VectorAssembler;
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
        data.cache();
        System.out.println();

        // correlation to Outcome
        System.out.println("Correlation to Outcome");
        for (String col : data.columns()) {
            if (!col.equals("Outcome")) {
                double corr = data.stat().corr("Outcome", col);
                System.out.println(col + " has correlation: " + corr);
            }
        }
        System.out.println();

        // create vector assembler
        System.out.println("Create vector assembler");
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{
                        "Pregnancies",
                        "Glucose",
                        "BloodPressure",
                        "SkinThickness",
                        "Insulin",
                        "BMI",
                        "DiabetesPedigreeFunction",
                        "Age"})
                .setOutputCol("features");
        // transform data
        Dataset<Row> output = assembler.transform(data);

        // print schema
        output.printSchema();

        // show
        output.show();

        Dataset<Row> final_data = output.select("features", "Outcome");
        final_data.printSchema();

        // add random split 0.7, 0,3: train, test = final_data.randomSplit([0.7, 0.3])
        // and create models logistic regression label col = Outcome: models = LogisticRegression(labelCol="Outcome")
        // model = models.fit(train)
        Dataset<Row>[] splits = final_data.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> train = splits[0];
        Dataset<Row> test = splits[1];

        LogisticRegression models = new LogisticRegression().setLabelCol("Outcome");
        LogisticRegressionModel model = models.fit(train);

        // Summary of model summary = model.summary
        System.out.println("Summary of model");
        model.summary().predictions().describe().show();

        // Evaluate model and test data

        // evaluate model
        System.out.println("Evaluate model");

        // predictions = model.evaluate(test)
        model.evaluate(test).predictions().show();
    }
}
