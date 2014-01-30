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

package org.apache.spark.mllib.classification;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

public class JavaNaiveBayesSuite implements Serializable {
  private transient JavaSparkContext sc;

  @Before
  public void setUp() {
    sc = new JavaSparkContext("local", "JavaNaiveBayesSuite");
  }

  @After
  public void tearDown() {
    sc.stop();
    sc = null;
    System.clearProperty("spark.driver.port");
  }

  private static final List<ClassLabeledPoint> POINTS = Arrays.asList(
    new ClassLabeledPoint("ClassA", new double[] {1.0, 0.0, 0.0}),
    new ClassLabeledPoint("ClassA", new double[] {2.0, 0.0, 0.0}),
    new ClassLabeledPoint("ClassB", new double[] {0.0, 1.0, 0.0}),
    new ClassLabeledPoint("ClassB", new double[] {0.0, 2.0, 0.0}),
    new ClassLabeledPoint("ClassC", new double[] {0.0, 0.0, 1.0}),
    new ClassLabeledPoint("ClassC", new double[] {0.0, 0.0, 2.0})
  );

  private int validatePrediction(List<ClassLabeledPoint> points, NaiveBayesModel model) {
    int correct = 0;
    for (ClassLabeledPoint p: points) {
      if (model.predict(p.features()).equals(p.classLabel())) {
        correct += 1;
      }
    }
    return correct;
  }

  @Test
  public void runUsingConstructor() {
    JavaRDD<ClassLabeledPoint> testRDD = sc.parallelize(POINTS, 2).cache();

    NaiveBayes nb = new NaiveBayes().setLambda(1.0);
    NaiveBayesModel model = nb.run(testRDD.rdd());

    int numAccurate = validatePrediction(POINTS, model);
    Assert.assertEquals(POINTS.size(), numAccurate);
  }

  @Test
  public void runUsingStaticMethods() {
    JavaRDD<ClassLabeledPoint> testRDD = sc.parallelize(POINTS, 2).cache();

    NaiveBayesModel model1 = NaiveBayes.train(testRDD.rdd());
    int numAccurate1 = validatePrediction(POINTS, model1);
    Assert.assertEquals(POINTS.size(), numAccurate1);

    NaiveBayesModel model2 = NaiveBayes.train(testRDD.rdd(), 0.5);
    int numAccurate2 = validatePrediction(POINTS, model2);
    Assert.assertEquals(POINTS.size(), numAccurate2);
  }
}
