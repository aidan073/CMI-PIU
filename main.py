from preprocess import Preprocessor

processor = Preprocessor("data/test.csv", "data/train.csv")
#processor.visualization()
processor.process(with_autoencoder=True)

