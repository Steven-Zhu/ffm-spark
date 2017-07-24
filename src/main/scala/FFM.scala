import breeze.linalg.{DenseVector => BDV}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer

/**
  * A spark version implementation of Field-aware Factorization Machines
  */
class FFM extends Serializable {

  def initModel(n: Int, m: Int, ffmParameter: FFMParameter): FFMModel = {
    val model: FFMModel = new FFMModel(n, m, ffmParameter.k, null, ffmParameter.normalization)
    val wLength = n * m * ffmParameter.k * 2
    model.W = new Array[Double](wLength)
    val coef = 1.0 / math.sqrt(model.k)

    var wIndex = 0
    for (j <- 0 until n)
      for (f <- 0 until m) {
        for (d <- 0 until ffmParameter.k) {
          model.W(wIndex) = coef * math.random
          wIndex += 1
        }
        for (d <- ffmParameter.k until ffmParameter.k * 2) {
          model.W(wIndex) = 1
          wIndex += 1
        }
      }
    model
  }

  def ffmReadProblem(sc: SparkContext, path: String, repartition: Boolean = false, partitions: Option[Int] = None): FFMProblem = {
    val datas = sc.textFile(path)
    var parsedData = datas.map(
      line => {
        val tokens = line.split("\\s")
        val y = if (tokens(0).toInt > 0) 1.0; else -1.0
        val nodes = tokens.drop(1).map(_.split(":")).map(items => {
          new FFMNode(items(0).toInt, items(1).toInt, items(2).toDouble)
        })
        (y, nodes)
      }
    )
    if (repartition) {
      if (partitions.isEmpty) throw new Exception("partitions should not be None")
      parsedData = parsedData.repartition(partitions.get)
    }
    val m = parsedData.map(x => x._2.map(y => y.fie).max).reduce(_ max _) + 1
    val n = parsedData.map(x => x._2.map(y => y.fea).max).reduce(_ max _) + 1
    new FFMProblem(n, m, parsedData.count().toInt, parsedData)
  }

  def computeFFM(nodes: Array[FFMNode],
                 n: Int,
                 m: Int,
                 k: Int,
                 initWeights: Array[Double],
                 normalization: Boolean = true) = {
    val r = if (normalization) {
      val norm = nodes.indices.map(i => math.pow(nodes(i).v, 2)).sum
      1.0 / norm
    } else {
      1.0
    }

    val align0 = k * 2
    val align1 = m * align0
    var t = 0.0

    for (n1 <- nodes.indices) {
      val j1 = nodes(n1).fea
      val f1 = nodes(n1).fie
      val v1 = nodes(n1).v
      if (j1 < n && f1 < m) {
        for (n2 <- n1 + 1 until nodes.length) {
          val j2 = nodes(n2).fea
          val f2 = nodes(n2).fie
          val v2 = nodes(n2).v
          if (j2 < n && f2 < m) {
            val w1Index = j1 * align1 + f2 * align0
            val w2Index = j2 * align1 + f1 * align0
            val v = v1 * v2 * r //!!!!!! r
            for (d <- 0 until k) {
              val w1 = initWeights(w1Index + d)
              val w2 = initWeights(w2Index + d)
              t = t + (w1 * w2 * v)
            }
          }
        }
      }
    }
    t
  }

  def gradientFFM(nodes: Array[FFMNode],
                  n: Int,
                  m: Int,
                  k: Int,
                  initWeights: Array[Double],
                  kappa: Double = 0,
                  eta: Double = 0,
                  lambda: Double = 0,
                  normalization: Boolean = true) = {
    val r = if (normalization) {
      val norm = nodes.indices.map(i => math.pow(nodes(i).v, 2)).sum
      1.0 / norm
    } else {
      1.0
    }

    val align0 = k * 2
    val align1 = m * align0

    val gradients = new Array[Double](initWeights.length)

    for (n1 <- nodes.indices) {
      val j1 = nodes(n1).fea
      val f1 = nodes(n1).fie
      val v1 = nodes(n1).v
      if (j1 < n && f1 < m) {
        for (n2 <- n1 + 1 until nodes.length) {
          val j2 = nodes(n2).fea
          val f2 = nodes(n2).fie
          val v2 = nodes(n2).v
          if (j2 < n && f2 < m) {
            val w1Index = j1 * align1 + f2 * align0
            val w2Index = j2 * align1 + f1 * align0
            val v = v1 * v2 * r //!!!!!! r
            val kappav = kappa * v
            val wg1Index = w1Index + k
            val wg2Index = w2Index + k
            for (d <- 0 until k) {
              val w1 = initWeights(w1Index + d)
              val w2 = initWeights(w2Index + d)
              val wg1 = initWeights(wg1Index + d)
              val wg2 = initWeights(wg2Index + d)
              val g1 = lambda * w1 + kappav * w2
              val g2 = lambda * w2 + kappav * w1

              gradients(w1Index + d) = -eta / math.pow(wg1, 0.5) * g1
              gradients(w2Index + d) = -eta / math.pow(wg2, 0.5) * g2
              gradients(wg1Index + d) = g1 * g1
              gradients(wg2Index + d) = g2 * g2

            }
          }
        }
      }
    }
    gradients //Array[Double]
  }

  def updateFFM(nodes: Array[FFMNode],
                n: Int,
                m: Int,
                k: Int,
                initWeights: Array[Double],
                kappa: Double = 0,
                eta: Double = 0,
                lambda: Double = 0,
                normalization: Boolean = true) = {
    val r = if (normalization) {
      val norm = nodes.indices.map(i => math.pow(nodes(i).v, 2)).sum
      1.0 / norm
    } else {
      1.0
    }

    val align0 = k * 2
    val align1 = m * align0

    for (n1 <- nodes.indices) {
      val j1 = nodes(n1).fea
      val f1 = nodes(n1).fie
      val v1 = nodes(n1).v
      if (j1 < n && f1 < m) {
        for (n2 <- n1 + 1 until nodes.length) {
          val j2 = nodes(n2).fea
          val f2 = nodes(n2).fie
          val v2 = nodes(n2).v
          if (j2 < n && f2 < m) {
            val w1Index = j1 * align1 + f2 * align0
            val w2Index = j2 * align1 + f1 * align0
            val v = v1 * v2 * r //!!!!!! r
            val kappav = kappa * v
            val wg1Index = w1Index + k
            val wg2Index = w2Index + k
            for (d <- 0 until k) {
              var w1 = initWeights(w1Index + d)
              var w2 = initWeights(w2Index + d)
              var wg1 = initWeights(wg1Index + d)
              var wg2 = initWeights(wg2Index + d)
              val g1 = lambda * w1 + kappav * w2
              val g2 = lambda * w2 + kappav * w1

              wg1 = wg1 + g1 * g1
              wg2 = wg2 + g2 * g2
              w1 = w1 - eta / math.pow(wg1, 0.5) * g1
              w2 = w2 - eta / math.pow(wg2, 0.5) * g2

              initWeights(w1Index + d) = w1
              initWeights(w2Index + d) = w2
              initWeights(wg1Index + d) = wg1
              initWeights(wg2Index + d) = wg2
            }
          }
        }
      }
    }
    initWeights //Array[Double]
  }

  def train(tr: FFMProblem, param: FFMParameter, va: Option[FFMProblem] = None) = {
    val model: FFMModel = initModel(tr.n, tr.m, param)
    val autoStop = param.autostop && va.isDefined && va.get.l > 0
    val slices = tr.X.getNumPartitions
    val stochasticLossHistory = new ArrayBuffer[Double](param.iters)

    var initWeights = model.W
    var i = 0
    var bestVaLoss = Double.MaxValue
    var converged = false
    while (!converged && i < param.iters) {
      var bcWeights = tr.X.context.broadcast(initWeights)
      val (gSum, lSum, size) = (tr.X).sample(false, param.miniFraction, 42 + i)
        .treeAggregate(new Array[Double](bcWeights.value.length), 0.0, 0L)(
          seqOp = (c, v) => {
            val t = computeFFM(v._2, model.n, model.m, model.k, bcWeights.value)
            val expnyt = math.exp(-v._1 * t)
            val kappa = -v._1 * expnyt / (1 + expnyt)
            val trLoss = math.log(1 + expnyt)
            (
              (BDV(c._1) + BDV(gradientFFM(v._2, model.n, model.m, model.k, bcWeights.value, kappa, param.eta, param.lambda)))
                .toArray,
              c._2 + trLoss,
              c._3 + 1
              )
          },
          combOp = (c1, c2) => {
            ((BDV(c1._1) + BDV(c2._1)).toArray, c1._2 + c2._2, c1._3 + c2._3)
          })

//      initWeights = wSum.map(_ / size)
      initWeights = (BDV(initWeights) + BDV(gSum)).toArray
      stochasticLossHistory += lSum / size

      bcWeights = tr.X.context.broadcast(initWeights)
      if (va.isDefined && va.get.l > 0) {
        var vaLoss = va.get.X.treeAggregate(0.0)(
          seqOp = (c, v) => {
            val t = computeFFM(v._2, model.n, model.m, model.k, bcWeights.value)
            val expnyt = math.exp(-v._1 * t)
            math.log(1 + expnyt)
          },
          combOp = (c1, c2) => {
            c1 + c2
          }
        )
        vaLoss /= va.get.l
        if (autoStop) {
          if (vaLoss > bestVaLoss) converged = true
          else bestVaLoss = vaLoss
        }
      }
      println("iter:" + (i + 1) + ",tr_loss:" + lSum / size)
      i += 1
    }
    model.W = initWeights
    //    (initWeights, stochasticLossHistory.toArray)
    (model, stochasticLossHistory.toArray)
  }

  /**
    *
    * @param tr train set
    * @param va validate set, can be none
    * @param params tuning parameters
    * @return
    */
  def ffmTrainWithValidation(tr: FFMProblem,
                             va: Option[FFMProblem] = None,
                             params: FFMParameter): FFMModel = {
    train(tr, params, va)._1
  }

  /**
    *
    * @param tr train set
    * @param params tuning parameters
    * @return
    */
  def ffmTrain(tr: FFMProblem,
               params: FFMParameter): FFMModel = {
    ffmTrainWithValidation(tr, None, params)
  }

  def ffmPredcit(data: RDD[Array[FFMNode]], model: FFMModel): RDD[Double] = {
    data.map(line => computeFFM(line, model.n, model.m, model.k, model.W, model.normablization))
  }
}

class FFMNode(val fie: Int, val fea: Int, val v: Double) extends Serializable

class FFMProblem(val n: Int, val m: Int, val l: Int,
                 var X: RDD[(Double, Array[FFMNode])]) extends Serializable

class FFMModel(val n: Int, val m: Int, val k: Int, var W: Array[Double], val normablization: Boolean)
  extends Serializable

class FFMParameter(val eta: Double, val lambda: Double, val iters: Int, val k: Int,
                   val normalization: Boolean, val random: Boolean, val autostop: Boolean,
                   val miniFraction: Double = 0.1) extends Serializable

