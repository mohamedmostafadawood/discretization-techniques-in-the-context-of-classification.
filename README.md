# discretization-techniques-in-the-context-of-classification.

In this code, I am examining the utility of discretization techniques in the context of classification.

I am using the Iris dataset.
First: “Train.csv”, which represents the training dataset, and “Test.csv”, which represents the testing dataset. 
Each row in the files is represented with three columns: “PetalLengthCm”, “PetalWidthCm” and “Species”. The
first two columns represent the two dimensions of the data, while the third column represents the flower
type. The objective is to classify each data point (row) in the file “Test.csv” to one of the three types of
flowers (Setosa, Versicolor, or Virginica).


Goal of this work:
- Discretize each of the “PetalLengthCm” and “PetalWidthCm” columns of the “Train.csv” and
“Test.csv” to n discrete values. The two datasets should be discretized together.
- Compute the mean (average) of the discretized “PetalLengthCm” and “PetalWidthCm” for the
points belonging to each flower type in the training dataset.
- For each point in the discretized testing data, compute the Euclidean distance between the point
and the mean point of each flower type. Assign the test point to the flower type of the nearest mean
point.
- Compute the classification accuracy as the number of correctly classified test points divided by the
total number of test points.
