/* SimpleApp.java */
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;

public class SimpleApp {
  public static void main(String[] args) {
    String logFile = "/opt/spark/README.md"; // le fichier que l'on veut lire
    SparkSession spark = SparkSession.builder().appName("Simple Application").getOrCreate();
    spark.sparkContext().setLogLevel("ERROR");
    Dataset<String> logData = spark.read().textFile(logFile).cache();

    long numAs = logData.filter(s -> s.contains("a")).count();
    long numBs = logData.filter(s -> s.contains("b")).count();

    System.out.println("Lignes avec a: " + numAs + ", lignes avec b: " + numBs);

    spark.stop();
  }
}
