/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.classification

import scala.collection.mutable.Map
import scala.collection.mutable.HashMap

import org.jblas.DoubleMatrix

import org.apache.spark.{SparkContext, Logging}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.util.MLUtils

/**
 * Model for Naive Bayes Classifiers.
 *
 * @param pi Log of class priors, whose dimension is C.
 * @param theta Log of class conditional probabilities, whose dimension is CxD.
 */
class NaiveBayesModel(val pi: Map[String, Double], val theta: Map[String, Array[Double]])
  extends MultiClassClassificationModel with Serializable {
  def predict(testData: RDD[Array[Double]]): RDD[String] = testData.map(predict)

  // Combine likelihoods and prior distributions for each class for the given input and select
  // the class with the highest log posterior distribution
  def predict(testData: Array[Double]): String = {
    val dataMatrix = new DoubleMatrix(testData)
    theta.map {
      case (paramClassLabel, classLikelihoods) =>
        val classLikelihoodsMatrix = new DoubleMatrix(classLikelihoods)
        (paramClassLabel, (classLikelihoodsMatrix.transpose.mmul(dataMatrix).add(pi(paramClassLabel)).toArray.head))
    }.foldLeft(("", Double.NegativeInfinity))((b, a) => if (a._2 > b._2) (a._1, a._2) else (b._1, b._2))._1
  }
}

/**
 * Trains a Naive Bayes model given an RDD of `(label, features)` pairs.
 *
 * This is the Multinomial NB ([[http://tinyurl.com/lsdw6p]]) which can handle all kinds of
 * discrete data.  For example, by converting documents into TF-IDF vectors, it can be used for
 * document classification.  By making every vector a 0-1 vector, it can also be used as
 * Bernoulli NB ([[http://tinyurl.com/p7c96j6]]).
 */
class NaiveBayes private (var lambda: Double)
  extends Serializable with Logging
{
  def this() = this(1.0)

  /** Set the smoothing parameter. Default: 1.0. */
  def setLambda(lambda: Double): NaiveBayes = {
    this.lambda = lambda
    this
  }

  /**
   * Run the algorithm with the configured parameters on an input RDD of ClassLabeledPoint entries.
   *
   * @param data RDD of (label, array of features) pairs.
   */
  def run(data: RDD[ClassLabeledPoint]) = {
    // Aggregates all sample points to driver side to get sample count and summed feature vector
    // for each label.  The shape of `zeroCombiner` & `aggregated` is:
    //
    //    label: Int -> (count: Int, featuresSum: DoubleMatrix)
    val zeroCombiner = Map.empty[String, (Int, DoubleMatrix)]
    val aggregated = data.aggregate(zeroCombiner)({ (combiner, point) =>
      point match {
        case ClassLabeledPoint(classLabel, features) =>
          val (count, featuresSum) = combiner.getOrElse(classLabel, (0, DoubleMatrix.zeros(1)))
          val fs = new DoubleMatrix(features.length, 1, features: _*)
          combiner += classLabel -> (count + 1, featuresSum.addi(fs))
      }
    }, { (lhs, rhs) =>
      for ((classLabel, (c, fs)) <- rhs) {
        val (count, featuresSum) = lhs.getOrElse(classLabel, (0, DoubleMatrix.zeros(1)))
        lhs(classLabel) = (count + c, featuresSum.addi(fs))
      }
      lhs
    })

    // Kinds of label
    val C = aggregated.size
    // Total sample count
    val N = aggregated.values.map(_._1).sum

    val pi = new HashMap[String, Double]
    val theta = new HashMap[String, Array[Double]]
    val piLogDenom = math.log(N + C * lambda)

    for ((label, (count, fs)) <- aggregated) {
      val thetaLogDenom = math.log(fs.sum() + fs.length * lambda)
      pi(label) = math.log(count + lambda) - piLogDenom
      theta(label) = fs.toArray.map(f => math.log(f + lambda) - thetaLogDenom)
    }

    new NaiveBayesModel(pi, theta)
  }
}

object NaiveBayes {
  /**
   * Trains a Naive Bayes model given an RDD of `(label, features)` pairs.
   *
   * This is the Multinomial NB ([[http://tinyurl.com/lsdw6p]]) which can handle all kinds of
   * discrete data.  For example, by converting documents into TF-IDF vectors, it can be used for
   * document classification.  By making every vector a 0-1 vector, it can also be used as
   * Bernoulli NB ([[http://tinyurl.com/p7c96j6]]).
   *
   * This version of the method uses a default smoothing parameter of 1.0.
   *
   * @param input RDD of `(label, array of features)` pairs.  Every vector should be a frequency
   *              vector or a count vector.
   */
  def train(input: RDD[ClassLabeledPoint]): NaiveBayesModel = {
    new NaiveBayes().run(input)
  }

  /**
   * Trains a Naive Bayes model given an RDD of `(label, features)` pairs.
   *
   * This is the Multinomial NB ([[http://tinyurl.com/lsdw6p]]) which can handle all kinds of
   * discrete data.  For example, by converting documents into TF-IDF vectors, it can be used for
   * document classification.  By making every vector a 0-1 vector, it can also be used as
   * Bernoulli NB ([[http://tinyurl.com/p7c96j6]]).
   *
   * @param input RDD of `(label, array of features)` pairs.  Every vector should be a frequency
   *              vector or a count vector.
   * @param lambda The smoothing parameter
   */
  def train(input: RDD[ClassLabeledPoint], lambda: Double): NaiveBayesModel = {
    new NaiveBayes(lambda).run(input)
  }

  def main(args: Array[String]) {
    if (args.length != 2 && args.length != 3) {
      println("Usage: NaiveBayes <master> <input_dir> [<lambda>]")
      System.exit(1)
    }
    val sc = new SparkContext(args(0), "NaiveBayes")
    val data = MLUtils.loadClassLabeledData(sc, args(1))
    val model = if (args.length == 2) {
      NaiveBayes.train(data)
    } else {
      NaiveBayes.train(data, args(2).toDouble)
    }
    println("Pi: " + model.pi.values.mkString("[", ", ", "]"))
    println("Theta:\n" + model.theta.values.map(_.mkString("[", ", ", "]")).mkString("[", "\n ", "]"))

    sc.stop()
  }
}
