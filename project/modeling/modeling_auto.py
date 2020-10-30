from collections import OrderedDict
from project.modeling.modeling_xlnet import XLNetForLMGCM
from project.modeling.modeling_distilbert import DistilBertForLMGCM
from transformers import (
    AlbertConfig,
    AutoConfig,
    BertConfig,
    DistilBertConfig,
    ElectraConfig,
    RobertaConfig,
    XLNetConfig,
    PretrainedConfig,
    AlbertForPreTraining,
    BertForPreTraining
)

MODEL_FOR_LMGC_M_MAPPING = OrderedDict(
    [
        # For Bert and Albert we just reuse their almost identical pre-training architecture
        (BertConfig, BertForPreTraining),
        (AlbertConfig, AlbertForPreTraining),
        # Else we use custom models with two heads
        (DistilBertConfig, DistilBertForLMGCM),
        # Note that XLNet requires permutational LM so it might require extra work with customizations
        (XLNetConfig, XLNetForLMGCM),
        # More models for LMGCM coming soon
        # (RobertaConfig, RobertaForLMGCM),
        # (ElectraConfig, ElectraForLMGCM),
    ]
)

class AutoModelForLMGCM:
    r"""
    This is a generic model class that will be instantiated as one of the model classes of the library---with a mlm and
    sequence classification head---when created with the when created with the
    :meth:`~transformers.AutoModelForLMGCM.from_pretrained` class method or the
    :meth:`~transformers.AutoModelForLMGCM.from_config` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoModelForLMGCM is designed to be instantiated "
            "using the `AutoModelForLMGCM.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelForLMGCM.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        for config_class, model_class in MODEL_FOR_LMGC_M_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_LMGC_M_MAPPING.keys()),
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs
            )

        for config_class, model_class in MODEL_FOR_LMGC_M_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_LMGC_M_MAPPING.keys()),
            )
        )
