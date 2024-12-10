from sklearn.impute import KNNImputer

class Preprocessor:
    def feature_engineering():
        pass
    def fill_nulls(samples:list[list]):
        imputer = KNNImputer(n_neighbors=5)
        transformed_samples = imputer.fit_transform(samples)
        
        return