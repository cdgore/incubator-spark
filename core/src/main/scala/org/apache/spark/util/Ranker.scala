package org.apache.spark.util

import scala.collection.mutable

import org.apache.spark.Accumulable
import org.apache.spark.AccumulableParam
import org.apache.spark.rdd.RDD
import org.apache.spark.serializer.JavaSerializer
import org.apache.spark.SparkConf

// Standard ranking assigns a rank to each row
private[spark]
class RankParam[R <: mutable.HashMap[T, Integer], T] extends AccumulableParam[R, T] {
  // Add a new element to the local set of rankings
  def addAccumulator(rankAccum: R, newElem: T): R = {
    rankAccum.put(newElem, rankAccum.size)
    rankAccum
  }
  
  // Merge two sets of rankings
  def addInPlace(t1: R, t2: R): R = {
    for (x <- t2) {
      t1.put(x._1, t1.size)
    }
    t1
  }

  def zero(initialValue: R): R = {
    // We need to clone initialValue, but it's hard to specify that R should also be Cloneable.
    // Instead we'll serialize it to a buffer and load it back.
    val ser = new JavaSerializer(new SparkConf(false)).newInstance()
    val copy = ser.deserialize[R](ser.serialize(initialValue))
    copy.clear()   // In case it contained stuff
    copy
  }
}

// Dense ranking assigns a unique rank to each unique element
private[spark]
class DenseRankParam[R <: mutable.HashMap[T, Integer], T] extends AccumulableParam[R, T] {
  // Add a new element to the local set of rankings
  def addAccumulator(rankAccum: R, newElem: T): R = {
    if (!rankAccum.contains(newElem))
      rankAccum.put(newElem, rankAccum.size)
    rankAccum
  }
  
  // Merge two sets of rankings
  def addInPlace(t1: R, t2: R): R = {
    for (x <- t2) {
      if (!t1.contains(x._1)) {
        t1.put(x._1, t1.size)
      }
    }
    t1
  }

  def zero(initialValue: R): R = {
    // We need to clone initialValue, but it's hard to specify that R should also be Cloneable.
    // Instead we'll serialize it to a buffer and load it back.
    val ser = new JavaSerializer(new SparkConf(false)).newInstance()
    val copy = ser.deserialize[R](ser.serialize(initialValue))
    copy.clear()   // In case it contained stuff
    copy
  }
}

object Ranker {
  def rank[T](unranked: RDD[T]) : RDD[(T, Integer)] = {
    val rankAccumulable = new Accumulable(new mutable.HashMap[T, Integer](), new RankParam[mutable.HashMap[T, Integer], T])
    unranked.foreach {
      x => rankAccumulable += x
    }
    unranked.sparkContext.parallelize(rankAccumulable.value.toSeq)
  }
}