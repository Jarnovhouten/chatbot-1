import streamlit as st
import json
import sagemaker
import boto3
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri

# Load credentials from Streamlit secrets
#aws_access_key = st.secrets["AWS_ACCESS_KEY_ID"]
#aws_secret_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
arn = st.secrets["ROLE_ARN"]
aws_region = st.secrets["AWS_DEFAULT_REGION"] 
hf_token = st.secrets["HF_TOKEN"]

# Configure AWS credentials
boto3.setup_default_session(
    region_name=aws_region
)

# Show title and description.
st.title("💬 Chatbot")
st.write(
    "This is a simple chatbot that uses Meta's Llama 3 model to generate responses. "
    "The model is deployed on AWS SageMaker for inference."
)

password = st.text_input("Password", type="password")
if not password == st.secrets["PASS"]:
    st.info("Please enter the password to continue.", icon="🗝️")
else:
    role = arn  # Use the ARN from secrets
    
    # Hub Model configuration
    hub = {
        "HF_MODEL_ID": "meta-llama/Meta-Llama-3-8B",
        "HF_NUM_CORES": "2",
        "HF_AUTO_CAST_TYPE": "fp16",
        "MAX_BATCH_SIZE": "4",
        "MAX_INPUT_TOKENS": "3686",
        "MAX_TOTAL_TOKENS": "4096",
        "HF_TOKEN": hf_token,
    }

    region = boto3.Session().region_name
    endpoint_name = "llama3-chatbot-endpoint"

    # Check if endpoint already exists
    sagemaker_client = boto3.client('sagemaker')
    try:
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        if response['EndpointStatus'] == 'InService':
            # If endpoint exists and is running, create predictor without deploying
            predictor = sagemaker.predictor.Predictor(
                endpoint_name=endpoint_name,
                sagemaker_session=sagemaker.Session()
            )
        else:
            # If endpoint exists but isn't in service, delete it and create new one
            sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            huggingface_model = HuggingFaceModel(
                image_uri=get_huggingface_llm_image_uri("huggingface",version="3.0.1"),
                env=hub,
                role=role,
            )
            predictor = huggingface_model.deploy(
                initial_instance_count=1,
                instance_type="ml.m5.xlarge",
                endpoint_name=endpoint_name
            )
    except sagemaker_client.exceptions.ClientError:
        # If endpoint doesn't exist, create it
        huggingface_model = HuggingFaceModel(
            image_uri=get_huggingface_llm_image_uri("huggingface",version="3.0.1"),
            env=hub,
            role=role,
        )
        predictor = huggingface_model.deploy(
            initial_instance_count=1,
            instance_type="ml.m5.xlarge",
            endpoint_name=endpoint_name
        )
    
    # Create a session state variable to store the chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display the existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create a chat input field
    if prompt := st.chat_input("What is up?"):
        # Store and display the current prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate a response using Llama 3
        response = predictor.predict({
            "inputs": prompt,
            "parameters": {
                "do_sample": True,
                "max_new_tokens": 128,
                "temperature": 0.7,
                "top_k": 50,
                "top_p": 0.95,
            }
        })

        # Display and store the response
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

