import torch
import torch.nn as nn
import transformers


class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict):
        """
        Setting MultitaskModel up as a PretrainedModel allows us
        to take better advantage of Trainer features
        """
        super().__init__(transformers.PretrainedConfig())

        self.taskmodels_dict = None
        ##########
        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)
        ##########

    @classmethod
    def create(cls, model_name, model_type_dict, model_config_dict):
        """
        This creates a MultitaskModel using the model class and config objects
        from single-task models. 

        We do this by creating each single-task model, and having them share
        the same encoder transformer.
        """
        taskmodels_dict = {}
        ##########
        shared_encoder = None
        taskmodels_dict = {}
        for task_name, model_type in model_type_dict.items():
            model = model_type.from_pretrained(
                model_name, 
                config=model_config_dict[task_name],
            )

            model_class_name = model.__class__.__name__
            if model_class_name.startswith("Bert"):
                encoder_name = "bert"
            elif model_class_name.startswith("Roberta"):
                encoder_name =  "roberta"
            elif model_class_name.startswith("Albert"):
                encoder_name = "albert"
            
            shared_encoder = getattr(model, encoder_name)
            taskmodels_dict[task_name] = model
        return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict)
        ##########
            
    def forward(self, task_name, **kwargs):
        return self.taskmodels_dict[task_name](**kwargs)
