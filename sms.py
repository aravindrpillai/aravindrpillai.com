import boto3
from property_reader import PropertyReader

def push_sms(message):
    key=PropertyReader.get_property("aws.sms.key")
    secret=PropertyReader.get_property("aws.sms.secret")
    region=PropertyReader.get_property("aws.sms.region")
    
    sns = boto3.client('sns', region_name=region, aws_access_key_id=key, aws_secret_access_key=secret)
    response = sns.publish(PhoneNumber='+91-9447020535',Message=message)
    
    print(response)