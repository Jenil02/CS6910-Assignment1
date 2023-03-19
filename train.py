# %%
import numpy as np
from tqdm import tqdm
import wandb
from keras.datasets import fashion_mnist

# %%
class Layer:
    def __init__(self, num_inputs, num_neurons, activation, weight_init):
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.activation_fn = activation
        self.weight_init = weight_init #what is this???
        self.w = np.random.randn(self.num_neurons, self.num_inputs)
        self.b = np.random.randn(self.num_neurons)
        if weight_init == 'Xavier':
            self.w = self.w/np.sqrt(self.num_inputs)
            self.b = self.b/np.sqrt(self.num_inputs)

    def activation(self,x):
        if self.activation_fn == 'ReLU':
            return np.maximum(0,x)
        if self.activation_fn == 'softmax':
            mx = np.max(x, axis = 1, keepdims=True)
            x -= mx
            # tp = np.sum(np.exp(x), axis=0, keepdims=True)
            # print(tp)
            return(np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True))
        if self.activation_fn == 'sigmoid':
            x = np.clip(x, -500, 500)
            return(1/(1+np.exp(-x)))
        if self.activation_fn == 'tanh':
            return np.tanh(x)

    def grad_activation(self, x):
        if self.activation_fn == 'ReLU':
            return 1*(x>0)
        if self.activation_fn == 'sigmoid':
            return (self.activation(x)*(1 - self.activation(x)))
        if self.activation_fn == 'tanh':
            return (1 - np.square(self.activation(x)))

    def forward(self, cur_input):
        re_bias = self.b.reshape(-1,1)
        self.a = np.dot(self.w,cur_input.T) + re_bias
        self.a = self.a.T
        self.h = self.activation(self.a)
        return self.h

    def backward(self, grad_a, prev_a, prev_h, grad_activation):
        self.dw = np.dot(grad_a.T, prev_h)
        self.db = np.sum(grad_a, axis=0)
        prev_h_grad = np.dot(grad_a, self.w)
        der = grad_activation(prev_a)
        grad_prev_a = prev_h_grad*der
        return grad_prev_a
        

# %%
class NeuralNetwork:
    def __init__(self, num_inputs, num_classes, num_hidden_layer, num_neurons, activation, weight_init):
        self.num_inputs = num_inputs
        self.num_classes = num_classes
        self.num_hidden_layer = num_hidden_layer
        self.num_neurons = num_neurons
        self.activation = activation
        self.weight_init = weight_init
        self.layers = []
        self.layers.append(Layer(num_inputs, num_neurons, activation, weight_init))
        for i in range(num_hidden_layer - 1):
            self.layers.append(Layer(num_neurons, num_neurons, 'ReLU', weight_init))
        self.layers.append(Layer(num_neurons, num_classes, 'softmax', weight_init))

    def forward(self, inputs):
        self.inputs = inputs
        cur_in = inputs
        for i in range(self.num_hidden_layer+1):
            cur_out = self.layers[i].forward(cur_in)
            cur_in = cur_out
        self.y_pred = cur_out
        return cur_out

    def backward(self, outputs):
        grad_a_L = -(outputs - self.y_pred)
        for i in range(self.num_hidden_layer, 0, -1):
            grad_a_L = self.layers[i].backward(grad_a_L,self.layers[i-1].a,self.layers[i-1].h, self.layers[i-1].grad_activation)

        self.layers[0].dw = np.dot(grad_a_L.T, self.inputs)
        self.layers[0].db = np.sum(grad_a_L, axis=0)
        
    def minibatch_sgd(self, dw, db, eta : float = 0.01, weight_decay : float = 0.0):
            for j in range(self.num_hidden_layer+1):
                self.layers[j].w -= eta*dw[j] + eta*weight_decay*self.layers[j].w
                self.layers[j].b -= eta*db[j] + eta*weight_decay*self.layers[j].b

    def momentum_gd(self, uw, ub, dw, db, eta : float = 0.01, weight_decay : float = 0.0, beta : float = 0.9):
        for j in range(self.num_hidden_layer+1):
            uw[j] = beta*uw[j] + dw[j]
            ub[j] = beta*ub[j] + db[j] 
            self.layers[j].w -= eta*uw[j] + eta*weight_decay*self.layers[j].w
            self.layers[j].b -= eta*ub[j] + eta*weight_decay*self.layers[j].b
        return uw, ub

    def NAG_gd(self, mw, mb, dw, db, eta : float = 0.01, weight_decay : float = 0.0, beta : float = 0.9):
        for j in range(self.num_hidden_layer+1):
            mw[j] = beta*mw[j] + dw[j]
            mb[j] = beta*mb[j] + db[j]
            self.layers[j].w -= eta*(beta*mw[j] + dw[j]) + eta*weight_decay*self.layers[j].w
            self.layers[j].b -= eta*(beta*mb[j] + db[j]) + eta*weight_decay*self.layers[j].b
        return mw, mb

    def RMSProp_gd(self, uw, ub, dw, db, eta : float = 0.01, weight_decay : float = 0.0, beta : float = 0.9, epsilon : float = 1e-8):
        for j in range(self.num_hidden_layer+1):
            uw[j] = beta*uw[j] + (1-beta)*dw[j]**2
            ub[j] = beta*ub[j] + (1-beta)*db[j]**2
            self.layers[j].w -= eta*dw[j]/(np.sqrt(uw[j])+epsilon) + eta*weight_decay*self.layers[j].w
            self.layers[j].b -= eta*db[j]/(np.sqrt(ub[j])+epsilon) + eta*weight_decay*self.layers[j].b
        return uw, ub
    
    def Adam_gd(self, mw, mb, uw, ub, dw, db, t, eta : float = 0.01, weight_decay : float = 0.0, beta1 : float = 0.9, beta2 : float = 0.999, epsilon : float = 1e-8):
        for j in range(self.num_hidden_layer+1):
            mw[j] = beta1*mw[j] + (1-beta1)*dw[j]
            mb[j] = beta1*mb[j] + (1-beta1)*db[j]
            uw[j] = beta2*uw[j] + (1-beta2)*(dw[j]**2)
            ub[j] = beta2*ub[j] + (1-beta2)*(db[j]**2)
            mw_hat = mw[j]/(1-beta1**t)
            mb_hat = mb[j]/(1-beta1**t)
            uw_hat = uw[j]/(1-beta2**t)
            ub_hat = ub[j]/(1-beta2**t)
            self.layers[j].w -= eta*mw_hat/(np.sqrt(uw_hat)+epsilon) + eta*weight_decay*self.layers[j].w
            self.layers[j].b -= eta*mb_hat/(np.sqrt(ub_hat)+epsilon) + eta*weight_decay*self.layers[j].b
        return mw, mb, uw, ub


    def NAdam_gd(self, mw, mb, uw, ub, dw, db, t, eta : float = 0.01, weight_decay : float = 0.0, beta1 : float = 0.9, beta2 : float = 0.999, epsilon : float = 1e-8):
        for j in range(self.num_hidden_layer+1):
            mw[j] = beta1*mw[j] + (1-beta1)*dw[j]
            mb[j] = beta1*mb[j] + (1-beta1)*db[j]
            uw[j] = beta2*uw[j] + (1-beta2)*dw[j]**2
            ub[j] = beta2*ub[j] + (1-beta2)*db[j]**2
            m_w_hat = mw[j]/(1-np.power(beta1, t+1))
            m_b_hat = mb[j]/(1-np.power(beta1, t+1))
            uw_hat = uw[j]/(1-np.power(beta2, t+1))
            ub_hat = ub[j]/(1-np.power(beta2, t+1))
            self.layers[j].w -= (eta/(np.sqrt(uw_hat) + epsilon))*(beta1*m_w_hat+ (1-beta1)*dw[j]/(1-np.power(beta1, t+1))) + eta*weight_decay*self.layers[j].w
            self.layers[j].b -= (eta/(np.sqrt(ub_hat) + epsilon))*(beta1*m_b_hat + (1-beta1)*db[j]/(1-np.power(beta1, t+1))) + eta*weight_decay*self.layers[j].b
        return mw, mb, uw, ub


    def train(self, X_train, y_train, X_test, y_test, batch_size, epochs, optimizer, eta : float = 0.001, weight_decay : float = 0.0, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
        for i in range(epochs):
            uw = [np.zeros_like(self.layers[j].w) for j in range(self.num_hidden_layer+1)]
            ub = [np.zeros_like(self.layers[j].b) for j in range(self.num_hidden_layer+1)]
            mw = [np.zeros_like(self.layers[j].w) for j in range(self.num_hidden_layer+1)]
            mb = [np.zeros_like(self.layers[j].b) for j in range(self.num_hidden_layer+1)]
            t = 1
            for i in tqdm(range(0, X_train.shape[0], batch_size)):
                x = X_train[i:i+batch_size]
                y = y_train[i:i+batch_size]
                self.forward(x)
                self.backward(y)
                dw = [self.layers[j].dw / X_train.shape[0] for j in range(self.num_hidden_layer+1)]
                db = [self.layers[j].db / X_train.shape[0] for j in range(self.num_hidden_layer+1)]
                if optimizer == "minibatch_sgd":
                    self.minibatch_sgd(dw, db, eta)
                elif optimizer == "momentum_gd":
                    uw, ub = self.momentum_gd(uw, ub,dw, db, eta, weight_decay, beta1)
                elif optimizer == "NAG_gd":
                    mw, mb = self.NAG_gd(mw, mb, dw, db, eta, weight_decay, beta1)
                elif optimizer == "RMSProp_gd":
                    uw, ub = self.RMSProp_gd(uw, ub, dw, db, eta, weight_decay, beta1, epsilon)
                elif optimizer == "Adam_gd":
                    mw, mb, uw, ub = self.Adam_gd(mw, mb, uw, ub, dw, db, t, eta, weight_decay, beta1, beta2, epsilon)
                elif optimizer == "NAdam_gd":
                    mw, mb, uw, ub = self.NAdam_gd(mw, mb, uw, ub, dw, db, t, eta, weight_decay, beta1, beta2, epsilon)   
                t += 1
            train_acc, train_loss = self.test(X_train, y_train)
            test_acc, test_loss = self.test(X_test, y_test)
            wandb.log({"train_acc": train_acc, "train_loss": train_loss, "val_acc": test_acc, "val_loss": test_loss})
                    

    def test(self, X_test, y_test):
        self.forward(X_test)
        y_pred = self.layers[-1].h
        loss = self.cross_entropy(y_pred, y_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_test, axis=1)

        return np.sum(y_pred == y_test)/y_test.shape[0], loss

    def cross_entropy(self, y_pred, y_true):
        return -np.sum(y_true*np.log(y_pred + 1e-9))/y_pred.shape[0]


# %%
# nn = NeuralNetwork(784, 10, 2, 128, 'ReLU', 'Xavier')
# (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# X_train = X_train.reshape(X_train.shape[0], -1) / 255
# X_test = X_test.reshape(X_test.shape[0], -1) / 255

# y_train = np.eye(10)[y_train]
# y_test = np.eye(10)[y_test]

# nn.train(X_train, y_train, X_test, y_test, 128, 2, "NAG_gd", 0.1)
# # nn.forward(X_train)


# %%

import argparse

parser = argparse.ArgumentParser()

# | Name | Default Value | Description |
# | :---: | :-------------: | :----------- |
# | `-wp`, `--wandb_project` | myprojectname | Project name used to track experiments in Weights & Biases dashboard |
# | `-we`, `--wandb_entity` | myname  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
# | `-d`, `--dataset` | fashion_mnist | choices:  ["mnist", "fashion_mnist"] |
# | `-e`, `--epochs` | 1 |  Number of epochs to train neural network.|
# | `-b`, `--batch_size` | 4 | Batch size used to train neural network. | 
# | `-l`, `--loss` | cross_entropy | choices:  ["mean_squared_error", "cross_entropy"] |
# | `-o`, `--optimizer` | sgd | choices:  ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] | 
# | `-lr`, `--learning_rate` | 0.1 | Learning rate used to optimize model parameters | 
# | `-m`, `--momentum` | 0.5 | Momentum used by momentum and nag optimizers. |
# | `-beta`, `--beta` | 0.5 | Beta used by rmsprop optimizer | 
# | `-beta1`, `--beta1` | 0.5 | Beta1 used by adam and nadam optimizers. | 
# | `-beta2`, `--beta2` | 0.5 | Beta2 used by adam and nadam optimizers. |
# | `-eps`, `--epsilon` | 0.000001 | Epsilon used by optimizers. |
# | `-w_d`, `--weight_decay` | .0 | Weight decay used by optimizers. |
# | `-w_i`, `--weight_init` | random | choices:  ["random", "Xavier"] | 
# | `-nhl`, `--num_layers` | 1 | Number of hidden layers used in feedforward neural network. | 
# | `-sz`, `--hidden_size` | 4 | Number of hidden neurons in a feedforward layer. |
# | `-a`, `--activation` | sigmoid | choices:  ["identity", "sigmoid", "tanh", "ReLU"] |

parser.add_argument("-wp", "--wandb_project", type=str, default="myprojectname", help="Project name used to track experiments in Weights & Biases dashboard")
parser.add_argument("-we", "--wandb_entity", type=str, default="myname", help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
parser.add_argument("-d", "--dataset", type=str, default="fashion_mnist", choices= ["mnist", "fashion_mnist"], help="Dataset used to train neural network.")

parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of epochs to train neural network.")

parser.add_argument("-b", "--batch_size", type=int, default=4, help="Batch size used to train neural network.")

parser.add_argument("-l", "--loss", type=str, default="cross_entropy", choices= ["mean_squared_error", "cross_entropy"], help="Loss function used to train neural network.")

parser.add_argument("-o", "--optimizer", type=str, default="sgd", choices= ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], help="Optimizer used to train neural network.")

parser.add_argument("-lr", "--learning_rate", type=float, default=0.1, help="Learning rate used to optimize model parameters")

parser.add_argument("-m", "--momentum", type=float, default=0.5, help="Momentum used by momentum and nag optimizers.")

parser.add_argument("-beta", "--beta", type=float, default=0.5, help="Beta used by rmsprop optimizer")

parser.add_argument("-beta1", "--beta1", type=float, default=0.5, help="Beta1 used by adam and nadam optimizers.")

parser.add_argument("-beta2", "--beta2", type=float, default=0.5, help="Beta2 used by adam and nadam optimizers.")

parser.add_argument("-eps", "--epsilon", type=float, default=0.000001, help="Epsilon used by optimizers.")

parser.add_argument("-w_d", "--weight_decay", type=float, default=.0, help="Weight decay used by optimizers.")

parser.add_argument("-w_i", "--weight_init", type=str, default="random", choices= ["random", "Xavier"], help="Weight initialization used to initialize model parameters.")

parser.add_argument("-nhl", "--num_layers", type=int, default=1, help="Number of hidden layers used in feedforward neural network.")

parser.add_argument("-sz", "--hidden_size", type=int, default=4, help="Number of hidden neurons in a feedforward layer.")

parser.add_argument("-a", "--activation", type=str, default="sigmoid", choices= ["identity", "sigmoid", "tanh", "ReLU"], help="Activation function used in feedforward neural network.")

args = parser.parse_args()

# %%
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], -1) / 255
X_test = X_test.reshape(X_test.shape[0], -1) / 255

y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

# %%
import wandb
from wandb.keras import WandbCallback

wandb.init(project=args.wandb_project, entity=args.wandb_entity)

config = wandb.config

config.epochs = args.epochs
config.batch_size = args.batch_size
config.loss = args.loss
config.optimizer = args.optimizer
config.learning_rate = args.learning_rate
config.momentum = args.momentum
config.beta = args.beta
config.beta1 = args.beta1
config.beta2 = args.beta2
config.epsilon = args.epsilon
config.weight_decay = args.weight_decay
config.weight_init = args.weight_init
config.num_layers = args.num_layers
config.hidden_size = args.hidden_size
config.activation = args.activation

# %%
nn = NeuralNetwork(input_size=784, output_size=10, num_layers=config.num_layers, hidden_size=config.hidden_size, activation=config.activation, weight_init=config.weight_init)

# %%
if config.optimizer == "minibatch_sgd":
    optimizer = nn.train(X_train, y_train, X_test, y_test, config.batch_size, config.epochs, "minibatch_gd", config.learning_rate, config.weight_decay)
elif config.optimizer == "momentum_gd":
    optimizer = nn.train(X_train, y_train, X_test, y_test, config.batch_size, config.epochs, "momentum_gd", config.learning_rate, config.weight_decay, config.momentum)
elif config.optimizer == "NAG_gd":
    optimizer = nn.train(X_train, y_train, X_test, y_test, config.batch_size, config.epochs, "NAG_gd", config.learning_rate, config.weight_decay, config.momentum)
elif config.optimizer == "RMSProp_gd":
    optimizer = nn.train(X_train, y_train, X_test, y_test, config.batch_size, config.epochs, "RMSProp", config.learning_rate, config.weight_decay, config.beta)
elif config.optimizer == "Adam_gd":
    optimizer = nn.train(X_train, y_train, X_test, y_test, config.batch_size, config.epochs, "Adam", config.learning_rate, config.weight_decay, config.beta1, config.beta2, config.epsilon)
elif config.optimizer == "NAdam_gd":
    optimizer = nn.train(X_train, y_train, X_test, y_test, config.batch_size, config.epochs, "NAdam", config.learning_rate, config.weight_decay, config.beta1, config.beta2, config.epsilon)

# %%
wandb.log({"loss": optimizer.loss, "accuracy": optimizer.accuracy})

# %%
wandb.finish()

