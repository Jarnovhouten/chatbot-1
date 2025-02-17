import sagemaker

def get_endpoint_status():
    try:
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        return response['EndpointStatus']
    except sagemaker_client.exceptions.ClientError:
        return None
