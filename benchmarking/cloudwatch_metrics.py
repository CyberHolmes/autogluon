from datetime import datetime, timedelta
import boto3
import csv
import time
import os.path
from pathlib import Path
# Click on Account Details and copy and paste the credentials into ~/.aws/credentials for AWS CLI then run the following code

cloudwatch = boto3.client('cloudwatch')

with open('autogluon_scheduler.log', 'r') as log:
    for line in log:
        word = line.split()
        start = datetime.strptime((word[2] + ' ' + word[3]), '%Y-%m-%d %H:%M:%S.%f')
        end = datetime.strptime((word[4] + ' ' + word[5]), '%Y-%m-%d %H:%M:%S.%f')
        response1 = cloudwatch.get_metric_statistics(
            Namespace='AWS/EC2',
            Dimensions=[
                {
                    'Name': 'InstanceId',
                    'Value': 'i-00f03a4ceb03b6d58'
                },
            ],
            MetricName='CPUUtilization',
            StartTime=start,
            EndTime=end,
            Period=5,
            Statistics=[
                'Average'
            ]
        )
        response2 = cloudwatch.get_metric_statistics(
            Namespace='AWS/EC2',
            Dimensions=[
                {
                    'Name': 'InstanceId',
                    'Value': 'i-06322a3de883d4cb5'
                },
            ],
            MetricName='CPUUtilization',
            StartTime=start,
            EndTime=end,
            Period=5,
            Statistics=[
                'Average'
            ]
        )
        Datapoints1 = response1['Datapoints']
        Datapoints2 = response2['Datapoints']
        # print(response['Datapoints'])

        cwd = os.getcwd()
        # print(cwd)
        cwd = os.path.join(cwd, word[0])
        # os.mkdir(cwd)
        cwd = os.path.join(cwd,word[1])
        # os.mkdir(cwd)
        Path(cwd).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(cwd,"Machine1.csv"), 'w') as csv_file:
            writer = csv.writer(csv_file)
            for datapoint in Datapoints1:
                writer.writerow([word[1], datapoint['Timestamp'], datapoint['Average']])
        with open(os.path.join(cwd,"Machine2.csv"), 'w') as csv_file:
            writer = csv.writer(csv_file)
            for datapoint in Datapoints2:
                writer.writerow([word[1], datapoint['Timestamp'], datapoint['Average']])

# os.path.join()
# name_of_file = raw_input("What is the name of the file: ")

# completeName = os.path.join(save_path, name_of_file+".txt")
# with open('dict.csv', 'r') as lines:
#     for line in lines:
#         word = line.split()
