import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class Dts {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder().appName("dts").getOrCreate();
        //Dataset<Row> df = spark.read().option("header","true").option("delimiter", ",").csv("./src/titanic.csv");
        java.util.Map< String, String> optionsMap = new java.util.HashMap< String, String>();
        optionsMap.put("delimiter",",");
        optionsMap.put("header","true");
        Dataset<Row> df = spark.read().options(optionsMap).csv("./src/titanic.csv");
        df.show();
        df.printSchema();
        df.select("Age", "Sex" ).show();
        Dataset<Row> mini_df = df.select(df.col("Age").cast("int"), df.col("Sex").cast("string"));    
        mini_df.printSchema();

        Dataset<Row> df2 = df.select(df.col("Survived").cast("boolean"),
        df.col("Pclass").cast("int"),
        df.col("Name").cast("string"),
        df.col("Sex").cast("string"),
        df.col("Age").cast("int"),
        df.col("Siblings/Spouses Aboard").cast("int"),
        df.col("Parents/Children Aboard").cast("int"),
        df.col("Fare").cast("double"));
        df2.printSchema();
        System.out.println("*****Distinct values for Pclass*****");
        df2.select("Pclass").distinct().show();
        long count = df2.select("Parents/Children Aboard").distinct().count();
        System.out.println("***********************************");
        System.out.println(count);
        System.out.println("***********************************");
        df2.describe().show();
        df2.groupBy("Survived").count().show();
        df2.groupBy("Survived", "Parents/Children Aboard").count().sort("Parents/Children Aboard", "Survived").show();
        df2.filter(df2.col("Age").leq(20)).groupBy("Survived").count().show();
        spark.close();
    }
}
