import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import autogluon.core as ag
import math

import torchvision
import torchvision.transforms as transforms
from tqdm.auto import tqdm
import argparse
from datetime import datetime
import time
import logging

import torchtext
from torchtext.datasets import text_classification
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
import os

parser = argparse.ArgumentParser()

parser.add_argument('--num_cpus', default=4, type=int, help='number of CPUs to use')
parser.add_argument('--num_gpus', default=0, type=int, help='number of GPUs to use')
parser.add_argument('--max_reward', default=90, type=int, help='convergence criterion')
parser.add_argument('--ip', default='ext_ips', help='additional ips to be added')

args = parser.parse_args()

class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

# Hyper Parameters to search over
@ag.args(
    lr=ag.space.Categorical(0.01, 0.02),
    wd=ag.space.Categorical(1e-4, 5e-4),
    epochs=ag.space.Categorical(5, 6),
    hidden_conv=ag.space.Categorical(6, 7),
    hidden_fc=ag.space.Categorical(80, 120),    
    batch_size=ag.space.Categorical(16, 32, 64, 128)
    ngrams = ag.space.Categorical(2,3)
)

class train_text_classification(args,reporter):
    # get variables from args
    lr = args.lr
    wd = args.wd
    epochs = args.epochs
    batch_size = args.batch_size
    ngrams = args.ngrams
    model = ConvNet(args.hidden_conv, args.hidden_fc)

    if not os.path.isdir('./.data'):
    os.mkdir('./.data')
    train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](
        root='./.data', ngrams=ngrams, vocab=None)

    VOCAB_SIZE = len(train_dataset.get_vocab())
    EMBED_DIM = 32
    NUN_CLASS = len(train_dataset.get_labels())
    model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUN_CLASS)

    # check if gpu is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    min_valid_loss = float('inf')

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    ###### Need to check how this scheduler works with the autogluon scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

    train_len = int(len(train_dataset) * 0.95)
    sub_train_, sub_valid_ = \
        random_split(train_dataset, [train_len, len(train_dataset) - train_len])

    def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label

    def train_func(sub_train_):

        # Train the model
        train_loss = 0
        train_acc = 0
        data = DataLoader(sub_train_, batch_size=args.batch_size, shuffle=True,
                        collate_fn=generate_batch)
        for i, (text, offsets, cls) in enumerate(data):
            optimizer.zero_grad()
            text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
            output = model(text, offsets)
            loss = criterion(output, cls)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            train_acc += (output.argmax(1) == cls).sum().item()

        # Adjust the learning rate
        scheduler.step()

        return train_loss / len(sub_train_), train_acc / len(sub_train_)

    def test(data_):
        loss = 0
        acc = 0
        data = DataLoader(data_, batch_size=args.batch_size, collate_fn=generate_batch)
        for text, offsets, cls in data:
            text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
            with torch.no_grad():
                output = model(text, offsets)
                loss = criterion(output, cls)
                loss += loss.item()
                acc += (output.argmax(1) == cls).sum().item()

        return loss / len(data_), acc / len(data_)

    for epoch in range(args.epochs):

        start_time = time.time()
        train_loss, train_acc = train_func(sub_train_)
        valid_loss, valid_acc = test(sub_valid_)

        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60

    test_loss, test_acc = test(test_dataset)

# define all the tasks
tasks = [
    train_text_classification,  # image classification task
]

# define run time table
run_times = []

# get external ips
ext_ips = open(args.ip, 'r').read().split('\n')


# Run every task with all available schedulers
for task in tasks:

    # define all schedulers
    if ext_ips[0] == '':
        schedulers = [
            ag.scheduler.OptimusScheduler(
                task,
                resource={'num_cpus': args.num_cpus, 'num_gpus': args.num_gpus},
                time_attr='epoch',
                reward_attr='accuracy',
                time_out=math.inf,
                max_reward=args.max_reward
            ),  # add the FIFO scheduler

            ag.scheduler.HyperbandScheduler(
                task,
                resource={'num_cpus': args.num_cpus, 'num_gpus': args.num_gpus},
                time_attr='epoch',
                reward_attr='accuracy',
                time_out=math.inf,
                max_reward=args.max_reward
            ),  # add the Hyperband scheduler

            ag.scheduler.RLScheduler(
                task,
                resource={'num_cpus': args.num_cpus, 'num_gpus': args.num_gpus},
                num_trials=10000,
                time_out=math.inf,
                time_attr='epoch',
                reward_attr='accuracy',
                max_reward=args.max_reward
            )  # add the FIFO scheduler
        ]
    else:
        schedulers = [
            ag.scheduler.OptimusScheduler(
                task,
                resource={'num_cpus': args.num_cpus, 'num_gpus': args.num_gpus},
                time_attr='epoch',
                reward_attr='accuracy',
                time_out=math.inf,
                max_reward=args.max_reward,
                dist_ip_addrs=ext_ips
            ),  # add the FIFO scheduler

            ag.scheduler.HyperbandScheduler(
                task,
                resource={'num_cpus': args.num_cpus, 'num_gpus': args.num_gpus},
                time_attr='epoch',
                reward_attr='accuracy',
                time_out=math.inf,
                max_reward=args.max_reward,
                dist_ip_addrs=ext_ips
            ),  # add the Hyperband scheduler

            ag.scheduler.RLScheduler(
                task,
                resource={'num_cpus': args.num_cpus, 'num_gpus': args.num_gpus},
                num_trials=10000,
                time_out=math.inf,
                time_attr='epoch',
                reward_attr='accuracy',
                max_reward=args.max_reward,
                dist_ip_addrs = ext_ips
            )  # add the FIFO scheduler
        ]

    # define the scheduler run time list
    scheduler_runtimes = []
    with open('autogluon_scheduler.log', 'w') as log:
        print('')

    for scheduler in schedulers:
        # run the task with selected scheduler
        # display the scheduler and available resources
        print('')
        print(scheduler)
        print('')

        # start the clock
        start_time = datetime.utcnow()

        # run the job with the scheduler
        scheduler.run()
        scheduler.join_jobs()

        # stop the clock
        stop_time = datetime.utcnow()

        # append to run times
        scheduler_runtimes.append([(stop_time - start_time).total_seconds(), scheduler.get_best_reward()])

        # publish to log
        with open('autogluon_scheduler.log', 'a') as log:
            print(task.__name__,scheduler.__class__.__name__, start_time, stop_time, file=log)

        # pause for a bit,before the next scheduler
        time.sleep(120)

print(scheduler_runtimes)