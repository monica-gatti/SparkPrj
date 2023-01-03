import java.util.HashMap;
import java.util.Map;
import org.apache.spark.sql.DataFrameNaFunctions;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.StandardScalerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import static org.apache.spark.sql.functions.col;

public class Penguin {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder().appName("penguin").getOrCreate();
        java.util.Map<String, String> options = new java.util.HashMap<String, String>();
        options.put("separator", ",");
        options.put("header", "true");
        Dataset<Row> df2 = spark.read().options(options).csv("./src/penguins_size.csv");
        df2.describe().show();
        df2.take(10);

        String[] col_names = df2.columns();
        int num_cols = col_names.length;
        Row RR = df2.describe().drop("summary").head();
        for (int i = 0; i < num_cols; i++) {
            System.out.println(col_names[i] + ": " + (df2.count()-Integer.parseInt(RR.getString(i))));
        }
        df2.select("species").distinct().show();
        df2.select("island").distinct().show();
        df2.select("sex").distinct().show();

        String[] str1 = {"culmen_length_mm"};
        String[] str2 = {"culmen_depth_mm"};
        String[] str3 = {"flipper_length_mm"};
        String[] str4 = {"body_mass_g"};
        //Dataset<Row> df3 = df2.na().fill(43.922, str1).na().fill(17.151, str2).na().fill(200.915, str3).na().fill(4201.754, str4);
        Dataset<Row> df3 = df2.na().fill(43.922, str1);
        df3.describe().show();
        df3.show();
        df3.groupBy("sex").count().show();
        Map<String, String> sex_replace = new HashMap<>();
        sex_replace.put("NA", "MALE");
        sex_replace.put(".", "MALE");
        df3 = df3.na().replace("sex", sex_replace);
        df3.select("sex").distinct().show();

        df3.createOrReplaceTempView("df3View");
        Dataset<Row> sqldf3 = spark.sql("SELECT body_mass_g FROM df3View");
        sqldf3.show();

        Dataset<Row> df4 = df3.sample(false, 0.1, 12345);
        df4.show();

        Dataset<Row> df3b = df3.select(
            col("species"),col("island"),
            col("culmen_length_mm").cast("double"),
            col("culmen_depth_mm").cast("double"),
            col("flipper_length_mm").cast("double"),
            col("body_mass_g").cast("double"), col("sex")
        );
        df3.write().csv("df3_after_clearing");
        StringIndexer indexer = new StringIndexer().setInputCols(new String[] {"species", "island", "sex"}).setOutputCols(new String[] {"speciesIndex", "islandIndex", "sexIndex"});
        Dataset<Row> indexed = indexer.fit(df3b).transform(df3b);
        indexed.show();
        indexed.describe().show();
        VectorAssembler assembler = new VectorAssembler().setInputCols(new String[] {"culmen_length_mm", "culmen_depth_mm","flipper_length_mm", "body_mass_g"}).setOutputCol("features");
        Dataset<Row> assembled = assembler.transform(df3b);
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

    }
    
}
