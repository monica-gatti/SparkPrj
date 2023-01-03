import static org.apache.spark.sql.functions.col;
import java.util.List;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.StandardScalerModel;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import scala.Tuple2;

public class Town {
    public static void main(String[] args) {
    
        SparkContext sc_raw = SparkSession.builder().getOrCreate().sparkContext();
        JavaSparkContext sc = JavaSparkContext.fromSparkContext(sc_raw);
        JavaRDD<String> rawRDD = sc.textFile("./src/housing.data");
        //comptez le nombre de villes près du fleuve et celles n'étant pas près du fleuve, col 3
        JavaRDD< String[]> RDD = rawRDD.map(line -> line.substring(1).split("\\s+"));
        System.out.println("Ville cote de la rivière Charles" + RDD.filter(line -> line[3].equals("1")).count());
        System.out.println("Ville que ne sont pas cote la riviere Charles " + RDD.filter(line -> line[3].equals("0")).count());
    

        //JavaPairRDD< String, Integer> pairsRiver = RDD.mapToPair(line -> new Tuple2(line[3], 1));
        //JavaPairRDD< String, Integer> countsRiver = pairsRiver.reduceByKey((a, b) -> a+b);
        //System.out.println("comptez le nombre de villes près du fleuve et celles n'étant pas près du fleuve:");
        //countsRiver.foreach(line -> System.out.println(line));
        //comptez par nombre d'occurrences le nombre moyen de pièces par logement, col 5
        JavaPairRDD<String, Integer> pairsRM = RDD.mapToPair(line -> new Tuple2(line[5], 1));
        // Regroupement des valeurs par clé (String = 6ème colonne) et incrémentation du compteur
        JavaPairRDD<String, Integer> countsRM = pairsRM.reduceByKey((a, b) -> a+b);
        //System.out.println("Il y a " + countsRM.count() + " valeurs distinctes dans la colonne RM");
        System.out.println("Valeurs de la colonne RM:");
        countsRM.mapToPair(x -> x.swap()).sortByKey(false).mapToPair(x -> x.swap()).take((int)countsRM.count()).forEach(line -> System.out.println(line));

        /*JavaRDD<DataPoint> dataRDD = rawRDD.map(line -> {
            String[] parts = line.substring(1).split("\\s+");// line.split("\\s+");
            DataPoint dp = new DataPoint();
            dp.setX(Double.parseDouble(parts[5]));
            return dp;
          });
        double sum = dataRDD.map(dp -> dp.getX()).reduce((x, y) -> x + y);
        long count = dataRDD.count();
        double avg = sum / count;
        System.out.println("comptez par nombre d'occurrences le nombre moyen de pièces par logement: "+ avg);
        */
        // affichez les différentes modalités de la variable RAD, col 8
        JavaRDD<String> RADColumnValues = rawRDD.map(line -> line.split("\\s+")[8]);
        JavaRDD<String> distinctRADValues = RADColumnValues.distinct();
        List<String> distinctRADList = distinctRADValues.collect();
        System.out.println("affichez les différentes modalités de la variable RAD");
        for (String value: distinctRADList) {
            System.out.println(value);
        }
       
        //Création d'un dataframe à partir du RDD 
        JavaRDD<String[]> rddOfArrays = rawRDD.map(line ->  line.substring(1).split("\\s+"));
        JavaRDD<Row> rddOfRows = rddOfArrays.map(fields -> RowFactory.create(fields));
        StructType houseSchema = DataTypes.createStructType(new StructField[] {
            DataTypes.createStructField("CRIM", DataTypes.StringType, true),
            DataTypes.createStructField("ZN", DataTypes.StringType, true),
            DataTypes.createStructField("INDUS", DataTypes.StringType, true),
            DataTypes.createStructField("CHAS", DataTypes.StringType, true),
            DataTypes.createStructField("NOX", DataTypes.StringType, true),
            DataTypes.createStructField("RM", DataTypes.StringType, true),
            DataTypes.createStructField("AGE", DataTypes.StringType, true),
            DataTypes.createStructField("DIS", DataTypes.StringType, true),
            DataTypes.createStructField("RAD", DataTypes.StringType, true),
            DataTypes.createStructField("TAX", DataTypes.StringType, true),
            DataTypes.createStructField("PTRATIO", DataTypes.StringType, true),
            DataTypes.createStructField("B", DataTypes.StringType, true),
            DataTypes.createStructField("LSTAT", DataTypes.StringType, true),
            DataTypes.createStructField("MEDV", DataTypes.StringType, true) });
        
        SparkSession spark = SparkSession.builder().appName("town").getOrCreate();
        Dataset<Row> dfhouse = spark.createDataFrame(rddOfRows, houseSchema);
        //Affichez le DataFrame nouvellement créé.
        System.out.println("Affichez le DataFrame nouvellement créé");
        dfhouse.printSchema();
        dfhouse.show();
        // Convertissez toutes les colonnes au format Double comme vu dans le cours.
        Dataset<Row> dfhouseDouble = dfhouse.select(
            col("CRIM").cast("double"),
            col("ZN").cast("double"),
            col("INDUS").cast("double"),
            col("CHAS").cast("double"),
            col("NOX").cast("double"),
            col("RM").cast("double"), 
            col("AGE").cast("double"),
            col("DIS").cast("double"),
            col("RAD").cast("double"),
            col("TAX").cast("double"),
            col("PTRATIO").cast("double"),
            col("B").cast("double"),
            col("LSTAT").cast("double"),
            col("MEDV").cast("double")
        );
        dfhouseDouble.printSchema(); 
        dfhouseDouble.show();
        //statistic avec filter, group by, count:
        System.out.println("Statistic filter");
        System.out.println("Stat 1");
        dfhouseDouble.filter(col("CRIM").geq(9.00)).groupBy("CHAS").count().show();
        /*
         * On vois que les areas avec plus de 9% de criminalite sont pas cote le fleve
        */
        System.out.println("Stat 2");
        Dataset<Row> dfWithPollution = dfhouseDouble.withColumn("POLLUETED",col("NOX").geq(0.50));
        dfWithPollution.filter(col("RAD").geq(4.00)).groupBy("POLLUETED").count().show();
        /*
         * On vois que les areas avec plus de 0.5 de RAD sont plus polluted
        */
        System.out.println("Stat 3");
        Dataset<Row> dfWithPoor = dfhouseDouble.withColumn("POOR",col("LSTAT").geq(20.00));
        dfWithPoor.filter(col("PTRATIO").geq(15.00)).groupBy("POOR").count().show();
        /*
         * On vois les areas que nous avons consideree povre ( bolean column POOR = true if LSTAT>20.00) ont moin de professeur par etudiant
        */
        //Affichez les statistiques numériques du DataFrame tel que la moyenne etc...
        System.out.println("Affichez les statistiques numériques du DataFrame tel que la moyenne etc...");
        dfhouseDouble.describe().show();

        //ML
        StringIndexer indexer = new StringIndexer()
        .setInputCols(new String[] {"MEDV"})
        .setOutputCols(new String[] {"MEDVINDEX"});
        Dataset<Row> indexed = indexer.fit(dfhouseDouble).transform(dfhouseDouble);
        indexed.show();
        indexed.describe().show();

        VectorAssembler assembler = new VectorAssembler()
        .setInputCols(new String[] {"CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO","B", "LSTAT"})
        .setOutputCol("features");
        Dataset<Row> assembled = assembler.transform(indexed);
        assembled.select("features").show();

        StandardScaler scaler = new StandardScaler()
        .setInputCol("features")
        .setOutputCol("scaledFeatures")
        .setWithStd(true)
        .setWithMean(false);
        StandardScalerModel scalerModel = scaler.fit(assembled);
        Dataset<Row> scaled = scalerModel.transform(assembled);
        scaled.show();
        scaled.describe().show();
        
        VectorAssembler assembler_fin = new VectorAssembler()
        .setInputCols(new String[] {"scaledFeatures"})
        .setOutputCol("house_features");
        Dataset<Row> data = assembler_fin.transform(scaled).select("MEDVINDEX","house_features");
        data = data.withColumnRenamed("MEDVINDEX", "label");
        data = data.withColumnRenamed("house_features", "features");
        data.show();

        LinearRegression lr = new LinearRegression();
        LinearRegressionModel lrModel = lr.fit(data);
        JavaRDD<Row> data_rdd = data.toJavaRDD();
        JavaPairRDD< Object, Object> predictionAndLabels = data_rdd.mapToPair(p ->
        new Tuple2<>(lrModel.predict(p.getAs(1)), p.getAs(0)));
        MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
        System.out.format("Weighted precision = %f\n", metrics.weightedPrecision());

        // Modèles et Pipeline 

        StringIndexer indexer2 = new StringIndexer().setInputCols(new String[] {"MEDVINDEX"})
        .setOutputCols(new String[] {"label"});
        Dataset<Row> indexed2 = indexer2.fit(dfhouseDouble).transform(dfhouseDouble);
  /*      VectorAssembler assembler2 = new VectorAssembler()
        .setInputCols(new String[] {"CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO","B", "LSTAT"})
        .setOutputCol("features_pre");
        Dataset<Row> data2 = assembler2.transform(indexed2).select("label", "features_pre");
        data2.printSchema();
        data2.show();

        StandardScaler scaler2 = new StandardScaler()
            .setInputCol("features_pre")
            .setOutputCol("features")
            .setWithStd(true)
            .setWithMean(false);
        LinearRegression lr2 = new LinearRegression();

        // Création pipeline
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] {scaler2, lr2});

        
        // Split Database en jeu de données (80%) et jeu de test (20%) - graine aléatoire = 12345
        Dataset<Row>[] data_split = data2.randomSplit(new double[] {0.8, 0.2}, 12345);
        Dataset<Row> train = data_split[0];
        Dataset<Row> test = data_split[1];
        
        // On entraine le pipeline sur le jeu d'entrainement
        PipelineModel model = pipeline.fit(train);
 
        // On effectue les prédictions sur le jeu de test
        Dataset<Row> predictions = model.transform(test);
        JavaRDD<Row> predictions_rdd = predictions.toJavaRDD();
        JavaPairRDD<Object, Object> predictionAndLabels2 = predictions_rdd.mapToPair(p ->
            new Tuple2<>(p.getAs(5), p.getAs(0)));
        MulticlassMetrics metrics2 = new MulticlassMetrics(predictionAndLabels2.rdd());
        System.out.format("Weighted precision = %f\n", metrics2.weightedPrecision());
*/
    }
    
}
