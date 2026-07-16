import numpy as np
import pandas as pd


class Predict:
    def __init__(self, dataX, modelList):
        self.dataX = dataX
        self.modelList = modelList

    def doPredict(self):
        predVectorList = []
        probVectorList = []

        for model in self.modelList:

            n_features = None

            if 'CatBoost' in type(model).__name__:
                try:
                    n_features = len(model.feature_names_)
                except Exception:
                    pass

            if n_features is None:
                for attr in ['n_features_in_', 'n_features_', 'n_features']:
                    if hasattr(model, attr):
                        n_features = getattr(model, attr)
                        break

            if n_features is None and hasattr(model, 'steps'):
                try:
                    estimator = model.steps[-1][1]
                    # 針對 Pipeline 內的 CatBoost
                    if 'CatBoost' in type(estimator).__name__:
                        try:
                            n_features = len(estimator.feature_names_)
                        except Exception:
                            pass

                    if n_features is None:
                        for attr in ['n_features_in_', 'n_features_', 'n_features']:
                            if hasattr(estimator, attr):
                                n_features = getattr(estimator, attr)
                                break
                except Exception:
                    pass

            if n_features is None or n_features == 0:
                n_features = self.dataX.shape[1]

            current_dataX = self.dataX.iloc[:, :n_features]

            try:
                predVector = model.predict(current_dataX)
                probVector = model.predict_proba(current_dataX)[:, 1]
            except Exception as e:
                try:
                    probVector = model.decision_function(current_dataX)
                except Exception:
                    probVector = predVector

            predVectorList.append(predVector)
            probVectorList.append(probVector)

        return predVectorList, probVectorList