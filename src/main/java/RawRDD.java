import org.apache.spark.api.java.*;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.rdd.RDD;
import org.apache.spark.api.java.JavaPairRDD;

import scala.Tuple2;
import scala.Tuple2$;

import java.util.Arrays;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;

public class RawRDD {

    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("RawRDD").setMaster("local");
        JavaSparkContext sc = new JavaSparkContext(conf);
        JavaRDD<String> rawRDD = sc.textFile("./src/housing.data");
        rawRDD.take(10).forEach(line -> System.out.println(line));
        
        System.out.println("*****************");
        System.out.println("*****************" + rawRDD.count() + "*****************");

        JavaRDD<String[]> RDD = rawRDD.map(line -> line.substring(1).split("\\s+"));
        RDD.take(10).forEach(line-> System.out.println(Arrays.toString(line)));


        JavaPairRDD<String, Integer> pairs = RDD.mapToPair(line -> new Tuple2(line[8], 1));
        pairs.foreach(line -> System.out.println(line));
        JavaPairRDD< String, Integer> counts = pairs.reduceByKey((a, b) -> a+b);
        counts.mapToPair(x -> x.swap()).sortByKey(false).mapToPair(x -> x.swap()).take(5).forEach(line -> System.out.println(line));
        System.out.println("**filter**");
        JavaPairRDD< String, Integer> filteredRDD =  RDD.filter(line -> line[1].equals("0.00")).mapToPair(line -> new Tuple2(line[8], 1));
        System.out.println("**fine filter**");
        sc.close(); 
        
        
    }


    
}
