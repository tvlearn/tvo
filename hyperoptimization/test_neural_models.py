import matplotlib.pyplot as plt
import torch as to
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from neural_models import FCnet, Deconvnet
from itertools import chain
import argparse
import logging
logging.disable()

parser=argparse.ArgumentParser()

parser.add_argument('--n_epochs', default=5, type=int)
parser.add_argument('--model', default='ConvDeConv', choices=['ConvDeConv', 'AE'], help='Model can be ConvDeConv or AE')
parser.add_argument('--lr', default=0.0001, type=float)

args=parser.parse_args()

# Script made with torch==1.7.0

# declare processor
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])

# prepare data
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = to.utils.data.DataLoader(mnist_trainset, batch_size=1000, shuffle=True)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = to.utils.data.DataLoader(mnist_testset, batch_size=1000, shuffle=True)
# define loss function
loss_fn = to.nn.MSELoss()

# define optimizer
optimizer = to.optim.Adam

class AutoEncoder(object):

    def __init__(self, shape=None, lr=0.003):
        if shape is None:
            shape = (28**2, 256, 64, 256, 28**2)

        # define weight initialization
        w_init = to.nn.init.xavier_normal_

        # initialize linear layers
        self.W0 = w_init(to.empty(shape[0], shape[1], requires_grad=True))
        self.W1 = w_init(to.empty(shape[1], shape[2], requires_grad=True))
        self.W2 = w_init(to.empty(shape[2], shape[3], requires_grad=True))
        self.W3 = w_init(to.empty(shape[3], shape[4], requires_grad=True))

        # load layers to optimizer
        self.optimizer = optimizer([self.W0, self.W1, self.W2, self.W3], lr=lr)
    def forward(self, x):
        # define activation function
        f = to.nn.functional.leaky_relu

        # forward pass
        h0  = f(x@self.W0)
        h1  = f(h0@self.W1)
        h2  = f(h1@self.W2)
        rec = to.tanh(h2@self.W3)

        return rec

    def train(self, epochs):

        losses = []

        for n_epoch in range(epochs):

            avg_train_loss = 0.
            no_datapoints = 0

            for batch, target in train_loader:

                # get 0s and 1s
                batch_ = batch[to.logical_or(target==0, target==1)].flatten(start_dim=1).double()
                if batch_.shape[0]==0:
                    continue

                # train
                self.optimizer.zero_grad()
                reconstruction = self.forward(batch_).resize_as(batch_)
                loss = loss_fn(reconstruction,batch_)
                loss.backward()
                self.optimizer.step()

                # log
                avg_train_loss += loss.data.item()
                no_datapoints += len(batch_)

            print('Epoch {} finished with average loss of {}'.format(n_epoch, round(avg_train_loss,6)))
            losses.append(avg_train_loss)

        print('Training is complete.')
        return losses


class FCAE(AutoEncoder):
    def __init__(self, shape=None):
        super().__init__(shape)
        n_stacks = len(shape)-1
        self.model = FCnet(
                            input_size=shape[0],
                            W_shapes=shape[1:],
                            fc_activations=['to.nn.LeakyReLU']*n_stacks,
                            dropouts=[0]*n_stacks,
                            dropout_rate=0.25
        )
        self.optimizer = optimizer(self.model.parameters(), lr=0.003)

    def forward(self, x):
        return self.model.forward(x)

class ConvDeconv(AutoEncoder):
    def __init__(self, shape=None, dtype=to.double, lr=0.003):
        super().__init__(shape,lr=lr)
        n_stacks = len(shape)-1
        half_stacks = n_stacks//2
        self.encoder = to.nn.Sequential(to.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, dtype=dtype),
                                        to.nn.Flatten(),
                                        to.nn.LeakyReLU(),
                                        to.nn.LazyLinear(out_features=shape[half_stacks], dtype=dtype),
                                        to.nn.LeakyReLU())

        self.decoder=Deconvnet(
            in_features=shape[half_stacks],
            filters_from_fc=1,
            kernels=None,
            output_shape=28**2,
            n_deconv_layers=2,
            n_kernels=[5,1],
            batch_norms=None,
            dc_activations=['to.nn.Tanh']*half_stacks,
        )
        self.decoder.D=28**2
        self.decoder.double()

        parameters=chain(self.encoder.parameters(), self.decoder.parameters())
        self.optimizer = optimizer(parameters, lr=0.003)

    def forward(self, x):
        z=self.encoder(x.double().reshape(x.shape[0],1,28,28))
        # set S_K(n) to 1 because we are reusing Deconvnet
        z=z.reshape(z.shape[0], 1, z.shape[-1])
        x_hat=self.decoder(z)

        return x_hat

plt.ioff()

# define model
if args.model=='ConvDeConv':
    model = ConvDeconv(shape=(28**2, 256, 64, 256, 28**2), lr=args.lr)
elif args.model=='AE':
    model = AutoEncoder(shape=(28**2, 256, 64, 256, 28**2), lr=args.lr)
# train model
avg_train_loss = model.train(epochs=args.n_epochs)

# get 0s and 1s for testing
for batch, target in test_loader:
    batch_ = batch[to.logical_or(target==0, target==1)].flatten(start_dim=1)
    if batch_.shape[0] != 0:
        break

# reconstruct
reconstruction = model.forward(batch_)

# plot originals against reconstruction
n_show = 7
fig, axs = plt.subplots(2, n_show)
for i in range(n_show):
    axs[0,i].imshow(batch_[i].reshape(28,28))
    axs[0,i].axis('off')
    axs[1,i].imshow(reconstruction[i].reshape(28,28).detach().numpy())
    axs[1,i].axis('off')
    if round((n_show-1)/2)==i:
        axs[0,i].title.set_text('Original')
        axs[1,i].title.set_text('Reconstruction')
plt.savefig('reconstruction')

# plot training loss
plt.figure()
plt.plot(avg_train_loss)
plt.title('loss')
plt.savefig("loss")
