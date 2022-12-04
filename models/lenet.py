import torch as t
from torch import nn
import numpy as np

params = {'model': 'lenet',
          'display_step': 250, 
          'batch_size': 256,
          'iterations': 3000, # 3000 or 500 
          'initial_lr': 0.05, 
          'lr_decay': 0.5,
          'adjust_lr_step': 1000,
          'initial_momentum': 0.9,
          'final_momentum': 0.95,
          'momentum_change_steps': 5_000,
          'adjust_momentum_step': 2_000,
          'apply_weight_norm': True,
          'weight_norm': 3.5,
          'adjust_norm_step': 1_000,
          'output_l2_decay': 0.001,
          'pooling': 'max',
          'activation':'tanh',
          'random_seed': 0}

class _LeNet(nn.Module):
    def __init__(self, layers):
      super(_LeNet, self).__init__()
      self.input_layer = nn.Sequential(*layers['input_layer'])
      self.hidden_layers = nn.Sequential(*layers['hidden_layers'])
      self.output_layer = nn.Sequential(*layers['output_layer'])
      self.activation = {}

    def layer_features(self, x):
      input_layer = self.input_layer(x)
      hidden = self.hidden_layers(input_layer)
      logits = self.output_layer(hidden)
      layer_features = {'input_layer': input_layer.data.cpu().numpy(),
                        'hidden': hidden.data.cpu().numpy(),
                        'logits': logits.data.cpu().numpy()}
      return layer_features
    
    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach().cpu().numpy()
        return hook

    def get_intermediate(self, x): 
        #TODO: complete this function to return the outputs of first conv and first pooling layer - Done
        # Returns the output as a numpy aray for ease of plotting it later
        self.input_layer[1].register_forward_hook(self.get_activation('conv1'))
        pool1_out = self.input_layer(x)  
        return self.activation['conv1'], pool1_out.detach().cpu().numpy()
    
    def forward(self, x):
        input_layer = self.input_layer(x)
        hidden = self.hidden_layers(input_layer)
        logits = self.output_layer(hidden)
        return logits

    def forward_threshold(self, x,  threshold=1e6):
        input_layer = self.input_layer(x)
        hidden = self.hidden_layers(input_layer)
        hidden = hidden.clip(max=threshold)
        logits = self.output_layer(hidden)
        return logits
    
    def forward_intermediate(self, x):
        input_layer = self.input_layer(x)
        hidden = self.hidden_layers(input_layer)
        return hidden

# We recommend you make this modules as the first component of your input layers 
class Reshape(t.nn.Module):
    def forward(self, x):
      return x.view(-1,3,32,32)

class LeNet(object):
    def __init__(self, params, use_custom=False):
        super().__init__()
        self.params = params
        self.device = t.device('cuda' if t.cuda.is_available() else 'cpu')
        self.activations = {'logistic': nn.Sigmoid(), 'relu': nn.ReLU(), 'tanh': nn.Tanh()}
        self.pooling = {'avg':nn.AvgPool2d(kernel_size=2, stride=2),'max':nn.MaxPool2d(kernel_size=2, stride=2)}
        # Instantiate model
        t.manual_seed(self.params['random_seed'])
        np.random.seed(self.params['random_seed'])
        layers = self._initialize_network() if not use_custom else self._initialize_network_custom()
        self.model = _LeNet(layers).to(self.device)

    def _initialize_network(self):
        # TODO: complete this function - Done
        # You may have to use Reshape() and nn.Flatten()
        # The architecture may be slightly different with no pooling
        if self.params['pooling'] != 'no':
            input_layers = [
                Reshape(),
                nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
                self.activations[self.params['activation']],
                self.pooling[self.params['pooling']]
            ]
            hidden_layers = [
                nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                self.activations[self.params['activation']],
                self.pooling[self.params['pooling']],
                # nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0),
                # self.activations[self.params['activation']],
                nn.Flatten(),
                nn.Linear(400, 120),
                self.activations[self.params['activation']],
                nn.Linear(120, 84),
                self.activations[self.params['activation']]
            ]
            output_layer = [
                nn.Linear(84, 10)
            ]
        else:
            input_layers = [
                Reshape(),
                nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
                self.activations[self.params['activation']]
            ]
            hidden_layers = [
                nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                self.activations[self.params['activation']],
                # nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0),
                # self.activations[self.params['activation']],
                nn.Flatten(),
                nn.Linear(9216, 120),
                self.activations[self.params['activation']],
                nn.Linear(120, 84),
                self.activations[self.params['activation']]
            ]
            output_layer = [
                nn.Linear(84, 10)
            ]
        layers = {'input_layer': input_layers,
          'hidden_layers': hidden_layers,
          'output_layer': output_layer}
        return layers
    
    def _initialize_network_custom(self):
        # Custom CNN network
        assert self.params['pooling'] != 'no'

        input_layers = [
            Reshape(),
            nn.Conv2d(3, 8, kernel_size=5, stride=1, padding='same'),
            self.activations[self.params['activation']],
            self.pooling[self.params['pooling']]
        ]
        hidden_layers = [
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding='same'),
            self.activations[self.params['activation']],
            self.pooling[self.params['pooling']],
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding='same'),
            self.activations[self.params['activation']],
            self.pooling[self.params['pooling']],
            nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=0),
            self.activations[self.params['activation']],
            nn.Flatten(),
            nn.Linear(128, 80),
            self.activations[self.params['activation']]
        ]
        output_layer = [
            nn.Linear(80, 10)
        ]

        layers = {'input_layer': input_layers,
          'hidden_layers': hidden_layers,
          'output_layer': output_layer}
        return layers
    
    def save_weights(self, path):
        t.save(self.model.state_dict(), path)

    def load_weights(self, path):
        self.model.load_state_dict(t.load(path,
                                          map_location=t.device(self.device)))

def lenet(pretrained=False, **kwargs):
    model_obj = LeNet(params, True)
    if pretrained:
        model_obj.load_weights('./weights/custom_lenet_tanh.pth')  
    return model_obj.model