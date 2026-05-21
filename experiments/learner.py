import small_text
from small_text import (
    PoolBasedActiveLearner,
    TransformerBasedClassificationFactory,
    TransformerModelArguments,
)

from nihdal import NIHDAL, NIHDAL_2, PretrainedDiscriminativeActiveLearning

DEFAULT_CLASSIFIER_KWARGS = {
    "device": "cuda",
    "mini_batch_size": 32,
    "num_epochs": 20,
    "lr": 5e-5,
    "class_weight": "balanced",
}


def _make_clf_factory(transformer_model_name, num_classes=2, classifier_kwargs=None):
    transformer_model = TransformerModelArguments(transformer_model_name)
    kwargs = dict(DEFAULT_CLASSIFIER_KWARGS)
    if classifier_kwargs:
        kwargs.update(classifier_kwargs)
    return TransformerBasedClassificationFactory(transformer_model, num_classes, kwargs=kwargs)


def _build_query_strategy(method, clf_factory, bias_indices=None):
    if method == "DAL":
        return PretrainedDiscriminativeActiveLearning(classifier_factory=clf_factory, num_iterations=10)
    if method == "NIHDAL":
        return NIHDAL(classifier_factory=clf_factory, num_iterations=10, bias_indices=bias_indices)
    if method == "NIHDAL_simon":
        return NIHDAL_2(classifier_factory=clf_factory, num_iterations=10, bias_indices=bias_indices)
    if method == "Random":
        return small_text.query_strategies.strategies.RandomSampling()
    if method == "Least Confidence":
        return small_text.LeastConfidence()
    if method == "Prediction Entropy":
        return small_text.PredictionEntropy()
    if method == "BALD":
        return small_text.query_strategies.bayesian.BALD()
    if method == "BADGE":
        return small_text.integrations.pytorch.query_strategies.strategies.BADGE(num_classes=2)
    if method == "Core Set":
        return small_text.query_strategies.coresets.GreedyCoreset()
    if method == "Contrastive":
        return small_text.query_strategies.strategies.ContrastiveActiveLearning()
    raise ValueError(f"Active Learning method {method!r} is unknown")


SUPPORTED_METHODS = [
    "Random",
    "Least Confidence",
    "Prediction Entropy",
    "BALD",
    "BADGE",
    "Core Set",
    "Contrastive",
    "DAL",
    "NIHDAL",
    "NIHDAL_simon",
]


def set_up_active_learner(transformer_model_name, active_learning_method, train,
                          bias_indices=None, classifier_kwargs=None):
    """Build a `PoolBasedActiveLearner` for the given method.

    The query strategy uses a separate classifier factory so the discriminative
    classifier doesn't share state with the task classifier.
    """
    clf_factory_task = _make_clf_factory(
        transformer_model_name, num_classes=2, classifier_kwargs=classifier_kwargs
    )
    clf_factory_discr = _make_clf_factory(
        transformer_model_name, num_classes=2, classifier_kwargs=classifier_kwargs
    )

    query_strategy = _build_query_strategy(
        active_learning_method, clf_factory_discr, bias_indices=bias_indices
    )

    return PoolBasedActiveLearner(
        clf_factory_task,
        query_strategy,
        train,
        reuse_model=False,
    )
