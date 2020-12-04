import math
import yaml
import time
import random
import torchtext
import argparse

import autogluon.core as ag
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchtext.data import get_tokenizer

from tqdm.auto import tqdm
from datetime import datetime


def generate_configuration_dict(config):
    """
    Creates a task dictionary

    config, full yaml file
    """

    config_dict = {}
    search_space_sizes = []

    for task in config['tasks']:
        configs = {}
        size = 1
        for parameter in config['tasks'][task]['parameters']:
            configs[parameter] = ag.space.Categorical(*config['tasks'][task]['parameters'][parameter])
            size = size * len(config['tasks'][task]['parameters'][parameter])
        config_dict[task] = configs
        search_space_sizes.append(size)

    return config_dict, search_space_sizes


def create_tasks(config):
    """
    Creates a list of tasks for benchmarking

    config, configuration file
    """

    with open(config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    config, search_space = generate_configuration_dict(config)

    # Hyper Parameters to search over
    @ag.args(
        **config['train_image_classification']
    )
    def train_image_classification(args, reporter):
        """
        args: arguments passed to the function through the ag.args designator
        reporter: The aug reporter object passed to the function by autogluon

        Reports the accuracy of the model to be monitored
        """

        def get_data_loaders(batch_size):
            """
            batch_size: The batch size of the dataset

            Returns the train and test data loaders
            """
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            train_data = torchvision.datasets.CIFAR10(
                root='./data',
                train=True,
                download=True,
                transform=transform
            )

            test_data = torchvision.datasets.CIFAR10(
                root='./data',
                train=False,
                download=True,
                transform=transform
            )

            train_data = torch.utils.data.DataLoader(
                train_data,
                batch_size=batch_size,
                shuffle=False
            )

            test_data = torch.utils.data.DataLoader(
                test_data,
                batch_size=batch_size,
                shuffle=False
            )

            return train_data, test_data

        class ConvNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 6, 5)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.fc1 = nn.Linear(16 * 5 * 5, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)

            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = x.view(-1, 16 * 5 * 5)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        # get variables from args
        lr = args.lr
        wd = args.wd
        epochs = 5
        batch_size = 16
        model = ConvNet()

        # get the data loaders
        train_loader, test_loader = get_data_loaders(batch_size)

        # check if gpu is available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)

        # if multiple GPUs are available, make it a data parallel model
        if device == 'cuda':
            model = nn.DataParallel(model)

        # get the loss function, and the optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=wd)

        # def the train function
        def train():
            """
            Trains the model
            """

            # set the model to train mode
            model.train()

            # run through all the batches in the dataset
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                # move the data to the target device
                inputs, targets = inputs.to(device), targets.to(device)

                # zero out gradients
                optimizer.zero_grad()

                # forward pass through the network
                outputs = model(inputs)

                # calculate loss and backward pass
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        def test(epoch):
            """
            epoch: epoch number

            Tests the model
            """

            # set the model to evaluation mode
            model.eval()

            # keep track of the test loss and correct predictions
            test_loss, correct, total = 0, 0, 0

            # stop tracking the gradients, reduces memory consumption
            with torch.no_grad():
                # run through the test set
                for batch_idx, (inputs, targets) in enumerate(test_loader):
                    # move the inputs to the target device
                    inputs, targets = inputs.to(device), targets.to(device)

                    # forward pass through the network
                    outputs = model(inputs)

                    # calculate the loss and labels
                    loss = criterion(outputs, targets)
                    test_loss += loss.item()

                    # keep track of the total correct predictions
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            # calculate the accuracy
            acc = 100. * correct / total
            # report the accuracy and the parameters used
            reporter(epoch=epoch, accuracy=acc, lr=lr, wd=wd, batch_size=batch_size)

        # run the testing and training script
        for epoch in tqdm(range(0, epochs)):
            train()
            test(epoch)

    @ag.args(
        **config['train_text_translation']
    )
    def train_text_translation(args, reporter):
        """
        args: arguments passed to the function through the ag.args designator
        reporter: The aug reporter object passed to the function by autogluon

        Reports the perplexity of the model to be monitored
        """

        class TransformerModel(nn.Module):

            def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
                super(TransformerModel, self).__init__()
                from torch.nn import TransformerEncoder, TransformerEncoderLayer
                self.model_type = 'Transformer'
                self.pos_encoder = PositionalEncoding(ninp, dropout)
                encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
                self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
                self.encoder = nn.Embedding(ntoken, ninp)
                self.ninp = ninp
                self.decoder = nn.Linear(ninp, ntoken)

                self.init_weights()

            def generate_square_subsequent_mask(self, sz):
                mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
                mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
                return mask

            def init_weights(self):
                initrange = 0.1
                self.encoder.weight.data.uniform_(-initrange, initrange)
                self.decoder.bias.data.zero_()
                self.decoder.weight.data.uniform_(-initrange, initrange)

            def forward(self, src, src_mask):
                src = self.encoder(src) * math.sqrt(self.ninp)
                src = self.pos_encoder(src)
                output = self.transformer_encoder(src, src_mask)
                output = self.decoder(output)
                return output

        class PositionalEncoding(nn.Module):

            def __init__(self, d_model, dropout=0.1, max_len=5000):
                super(PositionalEncoding, self).__init__()
                self.dropout = nn.Dropout(p=dropout)

                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0).transpose(0, 1)
                self.register_buffer('pe', pe)

            def forward(self, x):
                x = x + self.pe[:x.size(0), :]
                return self.dropout(x)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # define the text data
        TEXT = torchtext.data.Field(
            tokenize=get_tokenizer("basic_english"),
            init_token='<sos>',
            eos_token='<eos>',
            lower=True
        )

        def get_data():
            """
            returns train and validation data
            """

            # split into train, val, and test sets
            train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)

            # build the vocabulary
            TEXT.build_vocab(train_txt)

            def batchify(data, bsz):
                """
                Batches the data and sends it to device

                data, data to be batched
                bsz, batch_size
                """
                data = TEXT.numericalize([data.examples[0].text])
                # Divide the dataset into bsz parts.
                nbatch = data.size(0) // bsz
                # Trim off any extra elements that wouldn't cleanly fit (remainders).
                data = data.narrow(0, 0, nbatch * bsz)
                # Evenly divide the data across the bsz batches.
                data = data.view(bsz, -1).t().contiguous()
                return data.to(device)

            batch_size = 20
            eval_batch_size = 10

            # batch the train and validation data
            train_data = batchify(train_txt, batch_size)
            val_data = batchify(val_txt, eval_batch_size)

            return train_data, val_data

        def get_batch(source, i):
            seq_len = min(bptt, len(source) - 1 - i)
            data = source[i:i + seq_len]
            target = source[i + 1:i + 1 + seq_len].reshape(-1)
            return data, target

        train_data, val_data = get_data()

        bptt = 35
        ntokens = len(TEXT.vocab.stoi)  # the size of vocabulary
        emsize = 200  # embedding dimension
        nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = args.nlayers  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead = args.nhead  # the number of heads in the multiheadattention models
        dropout = args.dropout  # the dropout value
        model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

        criterion = nn.CrossEntropyLoss()
        lr = args.lr  # learning rate
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

        def train():
            model.train()  # Turn on the train mode

            src_mask = model.generate_square_subsequent_mask(bptt).to(device)
            for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
                data, targets = get_batch(train_data, i)
                optimizer.zero_grad()
                if data.size(0) != bptt:
                    src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
                output = model(data, src_mask)
                loss = criterion(output.view(-1, ntokens), targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        def evaluate(eval_model, data_source):
            eval_model.eval()  # Turn on the evaluation mode
            total_loss = 0.

            src_mask = model.generate_square_subsequent_mask(bptt).to(device)
            with torch.no_grad():
                for i in range(0, data_source.size(0) - 1, bptt):
                    data, targets = get_batch(data_source, i)
                    if data.size(0) != bptt:
                        src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
                    output = eval_model(data, src_mask)
                    output_flat = output.view(-1, ntokens)
                    total_loss += len(data) * criterion(output_flat, targets).item()

            total_loss = total_loss / (len(data_source) - 1)
            return math.exp(total_loss)

        epochs = 3  # The number of epochs

        for epoch in tqdm(range(0, epochs)):
            train()
            perplexity = -1 * evaluate(model, val_data)
            reporter(epoch=epoch, perplexity=perplexity, lr=lr, dropout=dropout, nhead=nhead, nlayers=nlayers)
            scheduler.step()

    task_list = [
        train_image_classification,
        train_text_translation
    ]

    return task_list, search_space


# define all the controllers
controllers = [
    'lstm',
    'alpha',
    'atten',
    'gru'
]

def create_schedulers(task, config, search_space):
    """
    creates the schedulers for the task using the config

    task, task to schedule
    config, configuration file
    search_space, search space size
    """

    with open(config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    num_cpus = config['num_cpus']
    num_gpus = config['num_gpus']
    max_reward = config['tasks'][task.__name__]['max_reward']
    reward_attr = config['tasks'][task.__name__]['reward_attr']
    dist_ips = config['dist_ips']

    schedulers = []

    for controller in controllers:

            scheduler_config = {
                'resource': {'num_cpus': num_cpus, 'num_gpus': num_gpus},
                'time_attr': 'epoch',
                'reward_attr': reward_attr,
                'time_out': math.inf,
                'num_trials': search_space,
                'max_reward': max_reward,
                'controller': controller
            }

            if dist_ips:
                scheduler_config['dist_ip_addrs'] = dist_ips

            scheduler = ag.scheduler.RLScheduler(task, **scheduler_config)
            schedulers.append(scheduler)

    return schedulers


if __name__ == "__main__":
    # load the commandline arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--conf', default='configuration.yaml', help='configuration file')
    parser.add_argument('--out', default='output/jct/rl_benchmark.csv', help='output file')
    parser.add_argument('-bootstrap', default=1, help='Number of times the experiment has been run')

    args = parser.parse_args()

    # create all the tasks
    tasks, search_space = create_tasks(args.conf)

    # set the seed
    seed = random.random()

    # shuffle the order of the controllers
    random.shuffle(controllers)

    with open(args.conf) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # count the number of machines
    if config['dist_ips']:
        num_machines = len(config['dist_ips'])
    else:
        num_machines = 1

    # create the results dataframe
    results = pd.DataFrame({
        'task': [],
        'controller': [],
        'sync': [],
        'runtime': [],
        'accuracy': [],
        'start_time': [],
        'end_time': [],
        'num_machines': [],
        'experiment': [],
        'search_space_size': []
    })

    # run the experiment multiple time
    for experiment in range(args.bootstrap):
        # schedule each task
        for tdx, task in enumerate(tasks):
            # experiment with different configurations of scheduler
            schedulers = create_schedulers(task, args.conf, search_space[tdx])
            # run the task with every scheduler
            for idx, scheduler in enumerate(schedulers):
                # start the clock
                start_time = datetime.utcnow()

                # run the job with the scheduler
                scheduler.run()
                scheduler.join_jobs()

                # stop the clock
                stop_time = datetime.utcnow()

                # add the experiment details to the results
                results = results.append({
                    'task': task.__name__,
                    'controller': scheduler.controller_type,
                    'runtime': (stop_time - start_time).total_seconds(),
                    'accuracy': scheduler.get_best_reward(),
                    'start_time': start_time,
                    'end_time': stop_time,
                    'num_machines': num_machines,
                    'experiment': experiment + 1,
                    'search_space_size': search_space[tdx]
                }, ignore_index=True)

                # sleep for 2 mins to help with cloudwatch
                time.sleep(120)

    # save the experiment details
    results.to_csv(args.out)
