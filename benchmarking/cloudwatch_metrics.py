from datetime import datetime, timedelta
import boto3
import csv
import time


# Click on Account Details and copy and paste the credentials into ~/.aws/credentials for AWS CLI then run the following code

cloudwatch = boto3.client('cloudwatch')


with open('autogluon_scheduler.log', 'r') as log:
    for line in log:
        word = line.split()
        start= datetime.strptime((word[1]+' '+word[2]), '%Y-%m-%d %H:%M:%S.%f')
        end = datetime.strptime((word[3]+' '+word[4]), '%Y-%m-%d %H:%M:%S.%f')
        response = cloudwatch.get_metric_statistics(
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
        Datapoints = response['Datapoints']

        print(response['Datapoints'])
        with open('dict.csv', 'a') as csv_file:
            writer = csv.writer(csv_file)
            for datapoint in Datapoints:
                writer.writerow([word[0],datapoint['Timestamp'], datapoint['Average']])

with open('dict,csv', 'r') as lines:
    for line in lines:
        word = line.split()

