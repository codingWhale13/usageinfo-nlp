import boto3

client = boto3.client('mturk') # endpoint_url='https://mturk-requester-sandbox.us-east-1.amazonaws.com'

with open('question.xml', 'r') as f:
    param = f.read()

response = client.create_hit_with_hit_type(
    HITTypeId="3BKZR1PB4UPWCDE3BFF22499LOQ6KB",
    MaxAssignments=2,
    LifetimeInSeconds=3600,
    Question=param,
    UniqueRequestToken='bsc_usage_option_test_1'
)