import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
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
    if (args.length > 2) pars = args(2).toInt

    val ffmProblem = ffm.ffmReadProblem(sc, args(0), true, Some(pars))
    val vaProblem = ffm.ffmReadProblem(sc, args(1), true, Some(pars))
    val va = vaProblem.X

    val model = ffm.ffmTrain(ffmProblem, new FFMParameter(0.5, 0.00002, 10, 4, true, true, false, 1))
    val validate = vaProblem.X.map(x => x._2)
    val pred = ffm.ffmPredcit(validate, model)
    val pairs = pred.zip( va.map(x => x._1))
    val biMetric = new BinaryClassificationMetrics(pairs, 10000)
    val auc = biMetric.areaUnderROC()
    println(s"auc: $auc")
  }
}
