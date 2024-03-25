



class _MultiTaskElasticNetPPM(GlmRegressorPPM):
    """
    PPM class for Elastic
    """

    def __init__(self, loo=True, n_alphas=100, l1_ratio = 0.0, standardize = False, **kwargs):
        super().__init__(ElasticNet(n_alphas=100, l1_ratio = 0.0), loo,**kwargs)

    
class MultiTaskRidgeRegressorPPM(_MultiTaskElasticNetPPM, GlmRegressorPPM,
                        PartialPredictionModelBase, ABC):
    """
    PPM class for regression that uses ridge as the GLM estimator.
    """
    ...

class MultiTaskRidgeRegressorPPM(_MultiTaskElasticNetPPM, GlmRegressorPPM,
                        PartialPredictionModelBase, ABC):
    """
    Ppm class for regression that uses ridge as the GLM estimator.
    """
    ...

class MultiTaskLassoRegressorPPM(_MultiTaskElasticNetPPM, GlmRegressorPPM,
                        PartialPredictionModelBase, ABC):
    """
    Ppm class for regression that uses lasso as the GLM estimator.
    """
    ...