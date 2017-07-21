import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by steven_zhu on 2017/7/18.
  */
object TestFFM {
  def main(args: Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf().setAppName("TESTFFM").setMaster("local[4]"))
    val ffm = new FFM()
    var pars = 10
    if (args.length > 1) args(1).toInt
    val ffmProblem = ffm.ffmReadProblem(sc, args(0), true, Some(pars))

    val splits = ffmProblem.X.randomSplit(Array(0.7,0.3))
    val (tr:RDD[(Double, Array[FFMNode])], va) = (splits(0), splits(1))
    val trp = new FFMProblem(ffmProblem.n, ffmProblem.m, tr.count().toInt, tr)

    val model = ffm.ffmTrain(trp, new FFMParameter(0.5, 0.00002, 20, 4, true, true, false, 1))
    ffm.ffmPredcit(ffmProblem, model)

  }
}
