# pipeline will go here

from preprocess import Preprocessor

processor = Preprocessor("data/test.csv", "data/train.csv")
#processor.visualization()
processor.process()