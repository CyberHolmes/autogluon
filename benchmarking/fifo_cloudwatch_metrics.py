from datetime import datetime, timedelta
import boto3
import csv
import time
import os.path
from pathlib import Path
import pandas as pd
import argparse
# Click on Account Details and copy and paste the credentials into ~/.aws/credentials for AWS CLI then run the following code
#scp -i "autogluon_arvind_gpu.pem" -r ubuntu@ec2-3-87-10-215.compute-1.amazonaws.com:~/autogluon/benchmarking/output/jct/ ./autog2

parser = argparse.ArgumentParser()
parser.add_argument('--out', default='output/resource_util/ag1_1', help='output file')

args = parser.parse_args()
if not os.path.exists(args.out):
    cwd = args.out
    Path(cwd).mkdir(parents=True, exist_ok=True)


jctlog = pd.read_csv("./output/jct/fifo_benchmark.csv")
cloudwatch = boto3.client('cloudwatch')

# create the results dataframe
results_cpu = pd.DataFrame({
    'task': [],
    'searcher': [],
    'runtime': [],
    'accuracy': [],
    'num_machines': [],
    'experiment': [],
    'search_space_size': [],
    'machine': [],
    'timestamp': [],
    'cpu_utilization': []
})
# create the results dataframe
results_gpu = pd.DataFrame({
    'task': [],
    'searcher': [],
    'runtime': [],
    'accuracy': [],
    'num_machines': [],
    'experiment': [],
    'search_space_size': [],
    'machine': [],
    'timestamp': [],
    'gpu_utilization': []
})
EC2_instances = ['i-0cb9872251019a481']
#EC2_instances = ['i-0750fccf339cbdf4f', 'i-07346a205ec87da05']
#EC2_instances = ['i-0cb9872251019a481',  'i-0750fccf339cbdf4f', 'i-07346a205ec87da05']

for i,ec2 in enumerate(EC2_instances):
    for j in range(jctlog.shape[0]):
        row = jctlog.iloc[[j]]
        starttime = row['start_time'].iloc[0]
        endtime = row['end_time'].iloc[0]

        response = cloudwatch.get_metric_statistics(
            Namespace='AWS/EC2',
            Dimensions=[
                {
                    'Name': 'InstanceId',
                    'Value': ec2
                },
            ],
            MetricName='CPUUtilization',
            StartTime=starttime,
            EndTime=endtime,
            Period=60,
            Statistics=[
                'Average'
            ]
        )
        response2 = cloudwatch.get_metric_statistics(
            Namespace='DeepLearningTrain',
            Dimensions=[
                {
                    'Name': 'InstanceId',
                    'Value': ec2
                },
            ],
            MetricName='GPU Usage',
            StartTime=starttime,
            EndTime=endtime,
            Period=60,
            Statistics=[
                'Average'
            ]
        )
        Datapoints1 = response['Datapoints']
        for datapoint in Datapoints1:
            results_cpu = results_cpu.append({
                'task': row['task'].iloc[0],
                'searcher': row['searcher'].iloc[0],
                'runtime': row['runtime'].iloc[0],
                'accuracy': row['accuracy'].iloc[0],
                'num_machines': row['num_machines'].iloc[0],
                'experiment': row['experiment'].iloc[0],
                'search_space_size': row['search_space_size'].iloc[0],
                'machine': i,
                'timestamp': datapoint['Timestamp'],
                'cpu_utilization': datapoint['Average']
            }, ignore_index=True)
        Datapoints2 = response2['Datapoints']
        for datapoint in Datapoints2:
            results_gpu = results_gpu.append({
                'task': row['task'].iloc[0],
                'searcher': row['searcher'].iloc[0],
                'runtime': row['runtime'].iloc[0],
                'accuracy': row['accuracy'].iloc[0],
                'num_machines': row['num_machines'].iloc[0],
                'experiment': row['experiment'].iloc[0],
                'search_space_size': row['search_space_size'].iloc[0],
                'machine': i,
                'timestamp': datapoint['Timestamp'],
                'gpu_utilization': datapoint['Average']
            }, ignore_index=True)
            # save the experiment details
results_cpu.to_csv(args.out+'/'+'fifo_cpu_util.csv')
results_gpu.to_csv(args.out+'/'+'fifo_gpu_util.csv')