import static org.apache.spark.sql.functions.col;

import java.util.Arrays;
import java.util.List;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.StandardScalerModel;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import scala.Tuple2;

public class Housing {
    public static void main(String[] args) {
    
        SparkContext sc_raw = SparkSession.builder().getOrCreate().sparkContext();
        JavaSparkContext sc = JavaSparkContext.fromSparkContext(sc_raw);
        JavaRDD<String> rawRDD = sc.textFile("./src/housing.data");
        //comptez le nombre de villes près du fleuve et celles n'étant pas près du fleuve, col 3
        JavaRDD< String[]> RDD = rawRDD.map(line -> line.substring(1).split("\\s+"));
        System.out.println("Ville cote de la rivière Charles" + RDD.filter(line -> line[3].equals("1")).count());
        System.out.println("Ville que ne sont pas cote la riviere Charles " + RDD.filter(line -> line[3].equals("0")).count());
        //comptez par nombre d'occurrences le nombre moyen de pièces par logement, col 5
        JavaPairRDD<String, Integer> pairsRM = RDD.mapToPair(line -> new Tuple2(line[5], 1));
        // Regroupement des valeurs par clé (String = 6ème colonne) et incrémentation du compteur
        JavaPairRDD<String, Integer> countsRM = pairsRM.reduceByKey((a, b) -> a+b);
        //System.out.println("Il y a " + countsRM.count() + " valeurs distinctes dans la colonne RM");
        System.out.println("Valeurs de la colonne RM:");
        countsRM.mapToPair(x -> x.swap()).sortByKey(false).mapToPair(x -> x.swap()).take((int)countsRM.count()).forEach(line -> System.out.println(line));
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
        
        SparkSession spark = SparkSession.builder().appName("Housing").getOrCreate();
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
        System.out.println("Statistic statistic avec filter, group by, count:");
        System.out.println("Stat 1: area with CRIM >= 9.00% cote et pas cote la fleve");
        dfhouseDouble.filter(col("CRIM").geq(9.00)).groupBy("CHAS").count().show();
        /*
         * On vois que les areas avec plus de 9% de criminalite sont pas cote le fleve
        */
        System.out.println("Stat 2: area with RAD >= 4, grouped by Pollueted or not pollueted (NOX >= 0.50)  ");
        Dataset<Row> dfWithPollution = dfhouseDouble.withColumn("POLLUETED",col("NOX").geq(0.50));
        dfWithPollution.filter(col("RAD").geq(4.00)).groupBy("POLLUETED").count().show();
        /*
         * On vois que les areas avec plus de 0.5 de RAD sont plus polluted
        */
        System.out.println("Stat 3: area with a professor for more than 15 students vs Poor - Not Poor according to LSTAT");
        Dataset<Row> dfWithPoor = dfhouseDouble.withColumn("POOR",col("LSTAT").geq(20.00));
        dfWithPoor.filter(col("PTRATIO").geq(15.00)).groupBy("POOR").count().show();
        /*
         * On vois les areas que nous avons consideree povre ( bolean column POOR = true if LSTAT>20.00) ont moin de professeur par etudiant
        */
        //Affichez les statistiques numériques du DataFrame tel que la moyenne etc...
        System.out.println("Affichez les statistiques numériques du DataFrame tel que la moyenne etc...");
        dfhouseDouble.describe().show();

        // Linear regression
        //L'agent immobilier veut predire le "valeur médiane des maisons habitées par ville en milliers de dollars"
        //Donc la label est la colonne "MEDV"
        dfhouseDouble = dfhouseDouble.withColumnRenamed("MEDV", "label");
        // Colonnes numériques in features
        VectorAssembler assembler = new VectorAssembler()
            .setInputCols(new String[] {"CRIM", "ZN", "NOX","RM", "AGE", "DIS", "PTRATIO", "B", "LSTAT"})
            .setOutputCol("features_predict")
            .setHandleInvalid("keep");
        Dataset<Row> data = assembler.transform(dfhouseDouble).select("label", "features_predict");
        data.printSchema();
        data.describe().show();
        data.select("features_predict").show();            
        // Suppression moyennes et mise à l'échelle des colonnes numériques
        StandardScaler scaler = new StandardScaler()
            .setInputCol("features_predict")
            .setOutputCol("features")
            .setWithStd(true)
            .setWithMean(false);
        LinearRegression lr = new LinearRegression();            
        // Création pipeline
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] {scaler, lr});
        // Split le jeu de données (80% train)
        Dataset<Row>[] data_split = data.randomSplit(new double[] {0.8, 0.2}, 12345);
        Dataset<Row> train = data_split[0];
        Dataset<Row> test = data_split[1];
        /* -------------------------------------------------- */
        // Validation croisée - essayer sur plus de regularization parameters
        // avec regression evaluation metric = RMSE
        // On cherche la parameter de regularization avec le minimum RMSE
        ParamMap[] paramGrid = new ParamGridBuilder() 
            .addGrid(lr.regParam(), new double[] {1, 0.1, 0.01, 0.2, 0.3, 0.4, 0.5})
            .build();
        RegressionEvaluator evaluator_rmse = new RegressionEvaluator()
            .setMetricName("rmse")
            .setLabelCol("label")
            .setPredictionCol("prediction");
        CrossValidator cv = new CrossValidator()
            .setEstimator(pipeline)
            .setEvaluator(evaluator_rmse)
            .setEstimatorParamMaps(paramGrid)
            .setNumFolds(4);
        CrossValidatorModel cvModel = cv.fit(train);
        Dataset<Row> predictions = cvModel.transform(test);
        predictions.printSchema();
        predictions.show();                        
        System.out.println(Arrays.toString(cvModel.avgMetrics()));                       
        /* Le plus petit RMSE est pour le  regularization parameter = 0.2 
        [1, 0.1, 0.01, 0.2, 0.3, 0.4, 0.5]
        [5.192943210444195, 5.173136584520603, 5.175342510287044, 5.172090819393381, 5.172229440284601, 5.173327312914106, 5.175216607782715]
        */
        spark.close();   

    }
}
