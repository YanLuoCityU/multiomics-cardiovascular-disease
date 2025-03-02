import math
import torch
import torch.nn as nn
import torch.nn.functional as F

'''

MLP-based architectures

Adapted from https://github.com/thbuerg/MetabolomicsCommonDiseases

'''
class SingleLayerNet(nn.Module):
    def __init__(self, input_dim=32, output_dim=2, final_activation=None, final_norm=False):
        super(SingleLayerNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if final_activation is not None and isinstance(final_activation, str):
            m = final_activation.split('.')
            final_activation = getattr(nn, m[1])
            print(final_activation)

        predictor_specs = [nn.Linear(self.input_dim, self.output_dim), ]
        if final_norm:
            predictor_specs.append(nn.BatchNorm1d(self.output_dim))
        if final_activation is not None:
            predictor_specs.append(final_activation())
        self.predictor = nn.Sequential(*predictor_specs)

    def forward(self, input):
        fts = self.predictor(input)
        return fts

class MLP(nn.Module):
    def __init__(self,
                 input_dim=32,
                 output_dim=2,
                 hidden_dim=256,
                 n_hidden_layers=None,
                 activation="nn.SiLU",
                 dropout_fn='nn.Dropout',
                 norm_fn='nn.BatchNorm1d',
                 norm_layer="all",
                 input_norm=False,
                 final_activation=None,
                 final_norm=False,
                 final_dropout=False,
                 snn_init=False,
                 dropout=0.5):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Dynamically evaluate functions if provided as strings
        activation = self._get_module_from_str(activation)
        final_activation = self._get_module_from_str(final_activation) if final_activation else None
        dropout_fn = self._get_module_from_str(dropout_fn)
        norm_fn = self._get_module_from_str(norm_fn)

        if isinstance(norm_layer, str) and norm_layer == "all":
            norm_layer = list(range(n_hidden_layers or len(hidden_dim)))

        self.input_norm = nn.LayerNorm(input_dim) if input_norm else None

        # Build the MLP
        layers = self._build_mlp(hidden_dim, n_hidden_layers, norm_fn, norm_layer, activation, dropout_fn)
        self.mlp = nn.Sequential(*layers)

        # Build the predictor
        predictor_layers = [nn.Linear(hidden_dim[-1] if isinstance(hidden_dim, (list, tuple)) else hidden_dim, output_dim)]
        if final_norm:
            predictor_layers.append(norm_fn(output_dim))
        if final_activation:
            predictor_layers.append(final_activation())
        if final_dropout:
            predictor_layers.append(dropout_fn(dropout))

        self.predictor = nn.Sequential(*predictor_layers)

        # Initialize parameters
        if snn_init:
            self._reset_parameters('predictor')
            self._reset_parameters('mlp')

    def forward(self, input):
        if self.input_norm:
            input = self.input_norm(input)
        fts = self.mlp(input)
        output = self.predictor(fts)
        return output

    def _get_module_from_str(self, module_str):
        """Utility to get a module from a string."""
        if isinstance(module_str, str):
            module_name, class_name = module_str.split('.')
            module_class = getattr(nn, class_name)
            return module_class
        return module_str

    def _build_mlp(self, hidden_dim, n_hidden_layers, norm_fn, norm_layer, activation, dropout_fn):
        """Utility to build the MLP layers."""
        layers = []
        hidden_dims = hidden_dim if isinstance(hidden_dim, (list, tuple)) else [hidden_dim] * n_hidden_layers

        for i, h in enumerate(hidden_dims):
            layers.append(nn.Linear(self.input_dim if i == 0 else hidden_dims[i-1], h))
            if i in norm_layer:
                layers.append(norm_fn(h))
            layers.append(activation())
            layers.append(dropout_fn(self.dropout))

        return layers

    def _reset_parameters(self, module_name):
        """Custom weight initialization."""
        for layer in getattr(self, module_name):
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=1 / math.sqrt(layer.out_features))
                if layer.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(layer.bias, -bound, bound)


class TaskSpecificMLP(nn.Module):
    def __init__(self,
                 skip_connection_mlp=MLP,
                 skip_connection_mlp_kwargs=dict(input_dim=32,
                                                 output_dim=None,
                                                 hidden_dim=None,
                                                 activation="nn.SiLU",
                                                 dropout_fn='nn.Dropout',
                                                 dropout=0.2,
                                                 final_activation="nn.SiLU",
                                                 final_norm=False,
                                                 final_dropout=False),
                 predictor_mlp=MLP,
                 predictor_mlp_kwargs=dict(input_dim=None,
                                           output_dim=None,
                                           hidden_dim=None,
                                           activation="nn.SiLU",
                                           dropout_fn='nn.Dropout',
                                           dropout=0.2,
                                           final_activation="nn.SiLU",
                                           final_norm=False,
                                           final_dropout=False),
                 ):
        super().__init__()
        self.skip_connection = skip_connection_mlp(**skip_connection_mlp_kwargs) # Skip connection MLP
        self.predictor = predictor_mlp(**predictor_mlp_kwargs) # Pooling MLP for final prediction

    def forward(self, features, covariates):
        skip_fts = self.skip_connection(covariates)
        h = torch.cat((features, skip_fts), dim=-1)
        out = self.predictor(h)
        return out

class OmicsNet(nn.Module):
    def __init__(self, 
                 outcomes_list=None,
                 shared_mlp_kwargs=dict(input_dim=None, 
                                        output_dim=512, 
                                        hidden_dim=[256, 256, 256], 
                                        norm_fn='nn.BatchNorm1d',
                                        norm_layer=[0],
                                        input_norm=False,
                                        final_norm=False,
                                        dropout_fn='nn.Dropout', 
                                        dropout=0.3,
                                        activation='nn.ReLU',
                                        final_activation='nn.ReLU',
                                        final_dropout=False),
                 skip_connection_mlp_kwargs=dict(input_dim=None,
                                                 output_dim=None,
                                                 hidden_dim=None,
                                                 activation="nn.ReLU",
                                                 dropout_fn='nn.Dropout',
                                                 dropout=0.2,
                                                 final_activation="nn.ReLU",
                                                 final_norm=False,
                                                 final_dropout=False),
                 predictor_mlp_kwargs=dict(input_dim=None,
                                           output_dim=None,
                                           hidden_dim=None,
                                           activation="nn.ReLU",
                                           dropout_fn='nn.Dropout',
                                           dropout=0.2,
                                           final_activation="nn.ReLU",
                                           final_norm=False,
                                           final_dropout=False)
                 ):
        super().__init__()
        self.outcomes_list = outcomes_list
        predictor_mlp_kwargs['input_dim'] = shared_mlp_kwargs['output_dim'] + skip_connection_mlp_kwargs['output_dim']
        self.shared_mlp = MLP(**shared_mlp_kwargs)
        
        # Define a separate ResidualHeadMLP for each outcome
        self.output_layers = nn.ModuleDict()
        for outcome in self.outcomes_list:
            self.output_layers[outcome] = TaskSpecificMLP(
                skip_connection_mlp=MLP,
                skip_connection_mlp_kwargs=skip_connection_mlp_kwargs,
                predictor_mlp=MLP,
                predictor_mlp_kwargs=predictor_mlp_kwargs
                )
        
    def forward(self, omics_data=None):
        outputs = {}
        shared_fts = self.shared_mlp(omics_data)
        for outcome in self.outcomes_list:
            outputs[outcome] = self.output_layers[outcome](shared_fts, omics_data)
        return outputs