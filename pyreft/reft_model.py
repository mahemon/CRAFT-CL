import pyvene as pv
import os 
import sys
import torch

def count_parameters(model):
    """Count parameters of a model that require gradients"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ReftModel(pv.IntervenableModel):
    """
    Base model for Reft methods.
    """
    def __init__(self, config, model, **kwargs):
        super().__init__(config, model, **kwargs)

    @staticmethod
    def _convert_to_reft_model(intervenable_model):
        reft_model = ReftModel(intervenable_model.config, intervenable_model.model)
        # Copy any other necessary attributes
        for attr in vars(intervenable_model):
            setattr(reft_model, attr, getattr(intervenable_model, attr))
        return reft_model

    @staticmethod
    def load(*args, **kwargs):
        model = pv.IntervenableModel.load(*args, **kwargs)
        return ReftModel._convert_to_reft_model(model)

    def save_intervention(self, save_directory, include_model=True):
        """
        Saves intervention artifacts (W, b, R) and optionally the model's trainable parameters.
        Verifies that learned_source.weight, learned_source.bias, and rotate_layer.parametrizations.weight.original exist.
        """
        os.makedirs(save_directory, exist_ok=True)
        print(f'save_directory: {save_directory}')

        for k, v in self.interventions.items():
            intervention = v
            binary_filename = f"intkey_{k}.bin"
            full_path = os.path.join(save_directory, binary_filename)

            if isinstance(intervention, pv.TrainableIntervention):
                state_dict = intervention.state_dict()
                torch.save(intervention.state_dict(), full_path)

        print(f'include_model: {include_model}')
        if include_model:
            model_state_dict = {
                n: p for n, p in self.model.named_parameters() if p.requires_grad
            }
            model_binary_filename = os.path.join(save_directory, "pytorch_model.bin")
            torch.save(model_state_dict, model_binary_filename)

    def save_pytorch_model(self, save_directory, include_model=True):
        """
        Verifies that learned_source.weight, learned_source.bias, and rotate_layer.parametrizations.weight.original exist.
        """
        os.makedirs(save_directory, exist_ok=True)
        if include_model:
            model_state_dict = {
                n: p for n, p in self.model.named_parameters() if p.requires_grad
            }
            model_binary_filename = os.path.join(save_directory, "pytorch_model.bin")
            torch.save(model_state_dict, model_binary_filename)

    def load_intervention(self, load_directory, include_model=True):
        """
        Instead of creating an new object, this function loads existing weights onto
        the current object. This is not a static method, and returns nothing.
        """
        for i, (k, v) in enumerate(self.interventions.items()):
            intervention = v
            binary_filename = f"intkey_{k}.bin"
            if isinstance(intervention, pv.TrainableIntervention):
                saved_state_dict = torch.load(os.path.join(load_directory, binary_filename))
                intervention.load_state_dict(saved_state_dict)
        if include_model:
            model_binary_filename = "pytorch_model.bin"
            saved_model_state_dict = torch.load(os.path.join(load_directory, model_binary_filename))
            self.model.load_state_dict(saved_model_state_dict, strict=False)
        
        return self.model

    
    def load_classifier_intervention(self, load_directory, include_model=True):
        # load model's trainable parameters as well
        if include_model:
            model_binary_filename = "pytorch_model.bin"
            saved_model_state_dict = torch.load(os.path.join(load_directory, model_binary_filename))
            self.model.load_state_dict(saved_model_state_dict, strict=False)
        return self.model

        
    def print_trainable_parameters(self):
        """
        Print trainable parameters.
        """
        _linked_key_set = set([])
        trainable_intervention_parameters = 0
        for k, v in self.interventions.items():
            if isinstance(v, pv.TrainableIntervention):
                if k in self._intervention_reverse_link:
                    if not self._intervention_reverse_link[k] in _linked_key_set:
                        _linked_key_set.add(self._intervention_reverse_link[k])
                        trainable_intervention_parameters += count_parameters(v)
                else:
                    trainable_intervention_parameters += count_parameters(v)

        trainable_model_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)

        all_model_parameters = sum(
            p.numel() for p in self.model.parameters())

        total_trainable_parameters = trainable_intervention_parameters + trainable_model_parameters
        
        print(
            f"trainable intervention params: {trainable_intervention_parameters:,d} || trainable model params: {trainable_model_parameters:,d}\n"
            f"model params: {all_model_parameters:,d} || trainable%: {100 * total_trainable_parameters / all_model_parameters}"
        )

