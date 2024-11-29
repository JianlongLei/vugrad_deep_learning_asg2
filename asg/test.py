import gc

from _context import vugrad
from _context import experiments

from question9 import MLP_ReLU
from experiments.train_mlp import MLP
import  vugrad as vg
from vugrad import load_mnist, load_synth, celoss
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import gc


# Function to train a model
def train_model(model, num_classes, xtrain, ytrain, xval, yval, args):
    num_instances, num_features = xtrain.shape
    print("num instances:", num_instances, "num features:", num_features, "num classes:", num_classes)
    mlp = model(input_size=num_features, output_size=num_classes)
    loss_result = []
    acc_result = []
    n, m = xtrain.shape
    b = args.batch_size
    print('\n## Starting training')
    for epoch in range(args.epochs):

        print(f'epoch {epoch:03}')

        ## Compute validation accuracy
        o = mlp(vg.TensorNode(xval))
        oval = o.value

        predictions = np.argmax(oval, axis=1)
        num_correct = (predictions == yval).sum()
        acc = num_correct / yval.shape[0]

        o.clear()  # gc the computation graph
        print(f'       accuracy: {acc:.4}')
        acc_result.append(acc)

        cl = 0.0  # running sum of the training loss

        # We loop over the data in batches of size `b`
        for fr in range(0, n, b):

            # The end index of the batch
            to = min(fr + b, n)

            # Slice out the batch and its corresponding target values
            batch, targets = xtrain[fr:to, :], ytrain[fr:to]

            # Wrap the inputs in a Node
            batch = vg.TensorNode(value=batch)

            outputs = mlp(batch)
            loss = vg.logceloss(outputs, targets)
            # -- The computation graph is now complete. It consists of the MLP, together with the computation of
            #    the scalar loss.
            # -- The variable `loss` is the TensorNode at the very top of our computation graph. This means we can call
            #    it to perform operations on the computation graph, like clearing the gradients, starting the backpropgation
            #    and clearing the graph.
            # -- Note that we set the MLP up to produce log probabilties, so we should compute the CE loss for these.

            cl += loss.value
            # -- We must be careful here to extract the _raw_ value for the running loss. What would happen if we kept
            #    a running sum using the TensorNode?

            # Start the backpropagation
            loss.backward()

            # pply gradient descent
            for parm in mlp.parameters():
                parm.value -= args.lr * parm.grad
                # -- Note that we are directly manipulating the members of the parm TensorNode. This means that for this
                #    part, we are not building up a computation graph.

            # -- In Pytorch, the gradient descent is abstracted away into an Optimizer. This allows us to build slightly more
            #    complexoptimizers than plain graident descent.

            # Finally, we need to reset the gradients to zero ...
            loss.zero_grad()
            # ... and delete the parts of the computation graph we don't need to remember.
            loss.clear()

        print(f'   running loss: {cl / n:.4}')
        loss_result.append(cl / n)
    return loss_result, acc_result


# Run the experiments
def run_experiments(dataset, lr=0.01):
    if dataset == "synth":
        (xtrain, ytrain), (xval, yval), num_classes = load_synth()
    else:
        (xtrain, ytrain), (xval, yval), num_classes = load_mnist(final=False, flatten=True)

    parser = ArgumentParser()

    parser.add_argument('-D', '--dataset',
                        dest='data',
                        help='Which dataset to use. [synth, mnist]',
                        default='synth', type=str)

    parser.add_argument('-b', '--batch-size',
                        dest='batch_size',
                        help='The batch size (how many instances to use for a single forward/backward pass).',
                        default=128, type=int)

    parser.add_argument('-e', '--epochs',
                        dest='epochs',
                        help='The number of epochs (complete passes over the complete training data).',
                        default=20, type=int)

    parser.add_argument('-l', '--learning-rate',
                        dest='lr',
                        help='The learning rate. That is, a scalar that determines the size of the steps taken by the '
                             'gradient descent algorithm. 0.1 works well for synth, 0.0001 works well for MNIST.',
                        default=lr, type=float)

    args = parser.parse_args()
    sigmoid_losses, sigmoid_accuracies = [], []
    relu_losses, relu_accuracies = [], []
    print(args)

    for _ in range(10):  # Run 10 times for each model
        # Sigmoid model
        s_loss, s_acc = train_model(MLP, num_classes, xtrain, ytrain, xval, yval, args)
        sigmoid_losses.append(s_loss)
        sigmoid_accuracies.append(s_acc)
    for _ in range(10):
        # ReLU model
        r_loss, r_acc = train_model(MLP_ReLU, num_classes, xtrain, ytrain, xval, yval,args)
        relu_losses.append(r_loss)
        relu_accuracies.append(r_acc)

    return sigmoid_losses, sigmoid_accuracies, relu_losses, relu_accuracies

def draw_plot_two_datasets(model_1, model_2, model_1_desc_label, model_2_desc_label, y_label, title, location):
    train_loss_array = np.array(model_1)
    val_loss_array = np.array(model_2)

    epochs = len(model_1[0]) if len(model_1) > 0 else 0
    t = np.arange(epochs)

    mu_train = train_loss_array.mean(axis=0)
    sigma_train = train_loss_array.std(axis=0)
    mu_val = val_loss_array.mean(axis=0)
    sigma_val = val_loss_array.std(axis=0)

    fig, ax = plt.subplots(1)

    ax.plot(t, mu_train, lw=2, label=model_1_desc_label, color='blue')
    ax.fill_between(t, mu_train + sigma_train, mu_train - sigma_train, facecolor='blue', alpha=0.3)

    ax.plot(t, mu_val, lw=2, label=model_2_desc_label, color='red')
    ax.fill_between(t, mu_val + sigma_val, mu_val - sigma_val, facecolor='red', alpha=0.3)

    ax.set_title(title)
    ax.legend(loc=location)
    ax.set_xlabel('Epochs')
    ax.set_ylabel(y_label)
    ax.grid()

    plt.show()
# Run and visualize results
if __name__ == "__main__":
    # s_losses, s_accs, r_losses, r_accs = run_experiments("synth")
    # # Plot Losses
    # draw_plot_two_datasets(s_losses, r_losses, "Sigmoid Loss", "ReLU Loss", "Loss", "Loss Comparison", 'upper right')
    #
    # # Plot Accuracies
    # draw_plot_two_datasets(s_accs, r_accs, "Sigmoid Accuracy", "ReLU Accuracy", "Accuracy", "Accuracy Comparison", 'lower right')

    s_losses, s_accs, r_losses, r_accs = run_experiments("mnist", lr=0.0001)
    # Plot Losses
    draw_plot_two_datasets(s_losses, r_losses, "Sigmoid Loss", "ReLU Loss", "Loss", "Loss Comparison", 'upper right')

    # Plot Accuracies
    draw_plot_two_datasets(s_accs, r_accs, "Sigmoid Accuracy", "ReLU Accuracy", "Accuracy", "Accuracy Comparison",
                           'lower right')



