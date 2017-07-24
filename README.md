# libffm-spark
A spark based implementation of FFM(Field-aware Factorization Machines).

The original C++ implementation is the project libffm(https://github.com/guestwalk/libffm).

## Usage
```scala
val sc = new SparkContext(...)
val ffm = new FFM()
val problem = ffm.ffmReadProblem(sc, trainPath)
val model = ffm.ffmTrain(problem, new Parameter(0.5, 0.00002, 10, 4, true, true, false, 1))

val validate = ffm.ffmReadProblem(sc, validPath)
val predict = model.ffmPredict(validate.X.map(x=>x._2), model)

```