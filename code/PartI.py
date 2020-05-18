import glob
import unicodedata
import string
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import classification_report
import numpy as np

all_trains = glob.glob('../data/cities_train/*.txt')
all_validations = glob.glob('../data/cities_val/*.txt')

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

print(all_trains)
print(all_validations)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

category_lines_train, category_lines_val = {}, {}
all_categories = []

def readLines(filename):
    lines = open(filename, 'r', encoding='utf-8', errors =  'ignore').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]


for filename in all_trains:
    category = filename.split('/')[-1].split('.')[0]
    if category not in all_categories:
        all_categories.append(category)
    lines = readLines(filename)
    category_lines_train[category] = lines


for filename in all_validations:
    category = filename.split('/')[-1].split('.')[0]
    if category not in all_categories:
        all_categories.append(category)
    lines = readLines(filename)
    category_lines_val[category] = lines

n_categories = len(all_categories)
print('n_categories =', n_categories)

def letter_to_tensor(letter):
    tensor = torch.zeros(1, n_letters)
    letter_index = all_letters.find(letter)
    tensor[0][letter_index] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        letter_index = all_letters.find(letter)
        tensor[li][0][letter_index] = 1
    return tensor

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))


n_hidden = 512
rnn = RNN(n_letters, n_hidden, n_categories)
# input = Variable(letter_to_tensor('A'))
# hidden = rnn.init_hidden()
#
# output, next_hidden = rnn(input, hidden)
# print('output.size =', output.size())

input = Variable(line_to_tensor('vilar do pinheiro'))
hidden = Variable(torch.zeros(1, n_hidden))

output, next_hidden = rnn(input[0], hidden)
# print(output)

# Preparing for training
def category_from_output(output):
    top_n, top_i = output.data.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

print(category_from_output(output))

def random_training_pair():
    category = random.choice(all_categories)
    line = random.choice(category_lines_train[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(line_to_tensor(line))
    return category, line, category_tensor, line_tensor

def random_validating_pair():
    category = random.choice(all_categories)
    line = random.choice(category_lines_val[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(line_to_tensor(line))
    return category, line, category_tensor, line_tensor

for i in range(10):
    category, line, category_tensor, line_tensor = random_training_pair()
    print('category = ', category, '/ line = ', line)

criterion = nn.NLLLoss()

learning_rate = 0.0015
optimizer = torch.optim.SGD(rnn.parameters(), lr = learning_rate)

def train(category_tensor, line_tensor):
    rnn.zero_grad()
    hidden = rnn.init_hidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    optimizer.step()

    return output, loss.item()

def validation(category_tensor, line_tensor):
    hidden = rnn.init_hidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)

    return loss.data

n_epochs = 100000
print_every = 5000
plot_every = 1000

# Keep track of losses for plotting
current_loss_train = 0
current_loss_val = 0
all_losses_train = []
all_losses_val = []

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()
'''
for epoch in range(1, n_epochs + 1):
    # Get a random training input and target
    category, line, category_tensor, line_tensor = random_training_pair()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print epoch number, loss, name and guess
    if epoch % print_every == 0:
        guess, guess_i = category_from_output(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, time_since(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if epoch % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

    plt.figure()
    plt.plot(all_losses)
    plt.show()
'''
for epoch in range(1, n_epochs + 1):
    # Get a random training input and target
    category, line, category_tensor, line_tensor = random_training_pair()
    output, loss = train(category_tensor, line_tensor)
    current_loss_train += loss

    category, line, category_tensor, line_tensor = random_validating_pair()
    loss = validation(category_tensor, line_tensor)
    current_loss_val += loss

    # Print epoch number, loss, name and guess
    if epoch % print_every == 0:
        guess, guess_i = category_from_output(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, time_since(start), loss, line, guess, correct))
    # Add current loss avg to list of losses
    if epoch % plot_every == 0:
        all_losses_train.append(current_loss_train / plot_every)
        all_losses_val.append(current_loss_val / plot_every)
        current_loss_train = 0
        current_loss_val = 0

plt.figure()
plt.plot(all_losses_train, color='skyblue', label='training loss')
plt.plot(all_losses_val, color='olive', label='validation loss')
plt.legend()

# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

# Just return an output given a line
def compute_acc(matrix):
    micro_acc = torch.sum(matrix.diag()) / torch.sum(matrix)
    macro_acc = TP_pos = 0
    for row in matrix:
        macro_acc += (row[TP_pos] / row.sum())
        TP_pos += 1
        macro_acc /= len(matrix)
    return float(micro_acc), float(macro_acc)

def evaluate(line_tensor):
    hidden = rnn.init_hidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

# Go through validation set and record which are correctly guessed
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = random_validating_pair()
    output = evaluate(line_tensor)
    guess, guess_i = category_from_output(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

micro_acc, macro_acc = compute_acc(confusion)
print('micro_acc:', micro_acc, 'macro_acc:', macro_acc)

# Normalize by dividing every row by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# Plot confusion matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.show()

def RNN_config(n_hidden=128):
    rnn = RNN(n_letters, n_hidden, n_categories)

    return rnn

# Keep track of losses for plotting
def train_config(n_epochs=100000):
    print_every = 5000
    plot_every = 1000
    current_loss_train = 0
    current_loss_val = 0
    all_losses_train = []
    all_losses_val = []
    for epoch in range(1, n_epochs + 1):
        # Get a random training input and target
        category, line, category_tensor, line_tensor = random_training_pair()
        output, loss = train(category_tensor, line_tensor)
        current_loss_train += loss

        category, line, category_tensor, line_tensor = random_validating_pair()
        loss = validation(category_tensor, line_tensor)
        current_loss_val += loss

        # Print epoch number, loss, name and guess
        if epoch % print_every == 0:
            guess, guess_i = category_from_output(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, time_since(start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if epoch % plot_every == 0:
            all_losses_train.append(current_loss_train / plot_every)
            all_losses_val.append(current_loss_val / plot_every)
            current_loss_train = 0
            current_loss_val = 0

    return all_losses_train, all_losses_val

def show_loss(all_losses_train, current_loss_val):
    plt.figure()
    plt.plot(all_losses_train, color='skyblue', label='training loss')
    plt.plot(all_losses_val, color='olive', label='validation loss')
    plt.legend()

def acc_score():
    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000

    # Just return an output given a line
    def evaluate(line_tensor):
        hidden = rnn.init_hidden()

        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)

        return output

    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = random_validating_pair()
        output = evaluate(line_tensor)
        guess, guess_i = category_from_output(output)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1

    micro_acc, macro_acc = compute_acc(confusion)
    print('micro_acc:', micro_acc, 'macro_acc:', macro_acc)

    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

lrs = [0.001, 0.0015, 0.002, 0.0025, 0.003]
for learning_rate in lrs:
    print('learning_rate = ', learning_rate)
    rnn = RNN_config(n_hidden=128)

    criterion = nn.NLLLoss()
    optimizer=torch.optim.SGD(rnn.parameters(), lr=learning_rate)

    print('all_losses_train, current_loss_val')
    all_losses_train, all_losses_val = train_config(n_epochs=100000)
    show_loss(all_losses_train, all_losses_val)

    print('acc_score')
    acc_score()


nhs = [128, 256, 384, 512, 640]
for n_hidden in nhs:
    print('n_hidden = ', n_hidden)
    rnn = RNN_config(n_hidden = n_hidden)

    criterion=nn.NLLLoss()
    optimizer=torch.optim.SGD(rnn.parameters(), lr=0.003)
    print('all_losses_train, current_loss_val')
    all_losses_train, all_losses_val = train_config(n_epochs=100000)
    show_loss(all_losses_train, all_losses_val)

    print('acc_score')
    acc_score()
