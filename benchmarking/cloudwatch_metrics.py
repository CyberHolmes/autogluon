from datetime import datetime, timedelta
import boto3
import csv

# Click on Account Details and copy and paste the credentials into ~/.aws/credentials for AWS CLI then run the following code

cloudwatch = boto3.client('cloudwatch')
response = cloudwatch.get_metric_statistics(
    Namespace='AWS/EC2',
    Dimensions=[
        {
            'Name': 'InstanceId',
            'Value': 'i-00f03a4ceb03b6d58'
        },
    ],
    MetricName='CPUUtilization',
    StartTime=datetime.utcnow() - timedelta(minutes=15),
    EndTime=datetime.utcnow(),
    Period=60,
    Statistics=[
        'Average'
    ]
)
Datapoints = response['Datapoints']

print(response['Datapoints'][1])

with open('dict.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for datapoint in Datapoints:
            writer.writerow([datapoint['Timestamp'], datapoint['Average']])
