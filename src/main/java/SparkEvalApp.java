
import static org.apache.spark.sql.functions.col;
import java.util.Arrays;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import scala.Tuple2;
public class SparkEvalApp {

    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("RawRDD").setMaster("local");
        // Création objet scala "sc"
        JavaSparkContext sc = new JavaSparkContext(conf);
        /* ---------------------------------------------------------------------
                a.- Lecture du fichier housing.data et découpage en colonnes    
           --------------------------------------------------------------------- */
        // Lecture (sc.textfile) et Chargement du fichier housing.data dans variable rawRDD
        JavaRDD<String> rawRDD = sc.textFile("./src/housing.data");    
        // Affichage des 10 premières lignes de rawRDD
        System.out.println("Affichage des 10 1eres lignes de la variable rawRDD :");
        rawRDD.take(10).forEach(line -> System.out.println(line));
        // Affichage du nombre total de lignes de rawRDD
        System.out.println("Il y a " + rawRDD.count() + " lignes dans le fichier housing.data");
        // Chargement du fichier housing.data dans variable tableau RDD avec split à chaque espace
        JavaRDD< String[]> RDD = rawRDD.map(line -> line.substring(1).split("\\s+"));
        /*  NB:
            L'utilisation d'un tableau de chaines de caractères "JavaRDD< String[]>"
            suppose l'import de la librarie "java.util.Arrays"  
        */            
        // Affichage des 10 premières lignes de RDD
        System.out.println("Affichage des 10 1eres lignes du tableau RDD :");
        RDD.take(10).forEach(line -> System.out.println(Arrays.toString(line)));
    /* ---------------------------------------------------------------------  
        b.- Comptage nbre de villes au bord de la rivière Charles ou éloignées (Colonne CHAS - Indice 3)
            0 = pas de frontière de la rivière Charles,
            1 = frontière de la rivière Charles
       --------------------------------------------------------------------- */

        System.out.println("Il y a " + RDD.filter(line -> line[3].equals("1")).count() + " ville(s) près de la rivière Charles");
        System.out.println("Il y a " + RDD.filter(line -> line[3].equals("0")).count() + " ville(s) non proche de la rivière Charles");

    /* ---------------------------------------------------------------------

        c.- Comptage par occurrences du nbre moyen de pièces par logement (Colonne RM - Indice 5)
            - Utilisation de la méthode ReduceByKey
       --------------------------------------------------------------------- */    
        /*
            Création d'un tupple "JavaPairRDD< String, Integer>"
                avec la 6ème colonne en clé (Indice 5) et 1 en valeur  
        */
        JavaPairRDD<String, Integer> pairsRM = RDD.mapToPair(line -> new Tuple2(line[5], 1));
        // Regroupement des valeurs par clé (String = 6ème colonne) et incrémentation du compteur
        JavaPairRDD<String, Integer> countsRM = pairsRM.reduceByKey((a, b) -> a+b);
        System.out.println("Il y a " + countsRM.count() + " valeurs distinctes dans la colonne RM");
        // liste des valeurs distinctes triées par ordre décroissant du nbre d'occurences
        System.out.println("Valeurs distinctes de la colonne RM, triées par ordre décroissant du nbre d'occurences :");
        countsRM.mapToPair(x -> x.swap()).sortByKey(false).mapToPair(x -> x.swap()).take((int)countsRM.count()).forEach(line -> System.out.println(line));
        /*
            --------------------------------------------------------------------------------
                d.- Comptage valeur disctinctes de la colonne RAD (indice tableau = 8)
                    - Utilisation de la méthode ReduceByKey        
            --------------------------------------------------------------------------------            
        */
        /*
            Création d'un tupple "JavaPairRDD<String, Integer>"
                avec la 9ème colonne en clé (Indice 8) et 1 en valeur  */
        JavaPairRDD<String, Integer> pairsRAD = RDD.mapToPair(line -> new Tuple2(line[8], 1));
        // Regroupement des valeurs par clé (String = 9ème colonne) et incrémentation du compteur
        JavaPairRDD<String, Integer> countsRAD = pairsRAD.reduceByKey((a, b) -> a+b);
        System.out.println("Il y a " + countsRAD.count() + " valeurs distinctes dans la colonne RAD");
        // liste des valeurs distinctes triées par ordre décroissant du nbre d'occurences
        System.out.println("Valeurs distinctes de la colonne RAD, triées par ordre décroissant du nbre d'occurences :");
        countsRAD.mapToPair(x -> x.swap()).sortByKey(false).mapToPair(x -> x.swap()).take((int)countsRAD.count()).forEach(line -> System.out.println(line));
        // -------------------------------------------------------------------------------
        //        Création d'un dataframe à partir du RDD          
        // -------------------------------------------------------------------------------
        // Create a SparkSession "Eval_Spark"
        SparkSession spark = SparkSession
            .builder()
            .appName("Eval_Spark")
            .getOrCreate();
        // Transformation RDD (JavaRDD<String[]>) en DataFrame (JavaRDD<Row>)
        JavaRDD<Row> rowRDD = RDD.map(line -> RowFactory.create(line));
        StructType schema = DataTypes.createStructType(new StructField[] {
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
        DataTypes.createStructField("MEDV", DataTypes.StringType, true)});

        // Création du dataframe à partir du RDD
        Dataset<Row> df = spark.createDataFrame(rowRDD, schema);
        // Visualisation du dataframe
        df.printSchema(); // Affichage du schéma du dataframe
        df.show();// Affichage des 20 premières lignes du dataframe
        //      New dataframe (à partir de df) avec conversion du type de colonnes
        // -------------------------------------------------------------------------------
        Dataset<Row> df2 = df.select(col("CRIM").cast("double"),
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
                                        col("MEDV").cast("double"));
        df2.printSchema(); // Affichage du schéma du dataframe
        // Affichage des statistiques descriptives du dataframe
        Dataset<Row> df3 = df2.select("CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE");
        df3.describe().show();
        Dataset<Row> df4 = df2.select("DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV");
        df4.describe().show();
        // Affichage du nbre de valeurs distinctes de la colonne RAD avec occurences
        System.out.println("Il y a " + df2.select("RAD").distinct().count() + " valeurs distinctes dans la colonne RAD");
        df2.groupBy("RAD").count().show();
        /* Affichage par proportion ratio profs/élèves selon le taux de pauvreté  */
        System.out.println("Affichage par proportion de ratio profs/élèves par ville");
        System.out.println(" (dont le taux de pauvreté est supérieur à 30%) : ");
        df2.filter(col("LSTAT").gt(30)).groupBy("PTRATIO").count().sort("PTRATIO").show();
        System.out.println("Affichage par proportion de ratio profs/élèves par ville");
        System.out.println(" (dont le taux de pauvreté est inférieur ou égal à 3%) : ");
        df2.filter(col("LSTAT").leq(3)).groupBy("PTRATIO").count().sort("PTRATIO").show();
        // -------------------------------------------------------------------------------
        //              Régression Logistique        
        // -------------------------------------------------------------------------------
        df2 = df2.withColumnRenamed("CRIM", "label");
        // Assemblage des colonnes numériques
        VectorAssembler assembler = new VectorAssembler()
            .setInputCols(new String[] {"ZN", "NOX","RM", "AGE", "DIS", "PTRATIO", "B", "LSTAT", "MEDV"})
            .setOutputCol("features_predict")
            .setHandleInvalid("keep");
        Dataset<Row> data = assembler.transform(df2).select("label", "features_predict");
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
        // Split Database en jeu de données (80%) et jeu de test (20%) - graine aléatoire = 12345
        Dataset<Row>[] data_split = data.randomSplit(new double[] {0.8, 0.2}, 12345);
        Dataset<Row> train = data_split[0];
        Dataset<Row> test = data_split[1];
        /* -------------------------------------------------- */
            // Validation croisée
        ParamMap[] paramGrid = new ParamGridBuilder() // Paramètres à tester
            .addGrid(lr.regParam(), new double[] {1, 0.1, 0.01})
            .build();
        RegressionEvaluator evaluator_rmse = new RegressionEvaluator()
            .setMetricName("rmse")
            .setLabelCol("label")
            .setPredictionCol("prediction");
        CrossValidator cv = new CrossValidator() // Validation croisée
            .setEstimator(pipeline)
            .setEvaluator(evaluator_rmse)
            .setEstimatorParamMaps(paramGrid)
            .setNumFolds(4);
        CrossValidatorModel cvModel = cv.fit(train); // On entraine le modèle
        // On effectue les prédictions
        Dataset<Row> predictions = cvModel.transform(test);
        predictions.printSchema();
        predictions.show();                        
        // score moyen obtenu par les différents modèles testés
        System.out.println(Arrays.toString(cvModel.avgMetrics()));                       
        // Fermeture de la session Spark
        spark.close();                                                  

    }

}
