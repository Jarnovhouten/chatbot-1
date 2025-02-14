import streamlit as st
import json
import sagemaker
import boto3
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri

# Load credentials from Streamlit secrets
try:
    aws_access_key = st.secrets["AWS_ACCESS_KEY_ID"]
    aws_secret_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
    aws_region = st.secrets["AWS_DEFAULT_REGION"] 
    hf_token = st.secrets["HF_TOKEN"]
except:
    st.error("Failed to load credentials from Streamlit secrets")
    st.stop()

# Configure AWS credentials
boto3.setup_default_session(
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=aws_region
)

# Show title and description.
st.title("💬 Chatbot")
st.write(
    "This is a simple chatbot that uses Meta's Llama 3 model to generate responses. "
    "The model is deployed on AWS SageMaker for inference."
)

password = st.text_input("Password", type="password")
if not password == "passwordamus":
    st.info("Please enter the password to continue.", icon="🗝️")
else:
    try:
        role = sagemaker.get_execution_role()
    except ValueError:
        iam = boto3.client("iam")
        role = iam.get_role(RoleName="sagemaker_execution_role")["Role"]["Arn"]

    # Hub Model configuration. https://huggingface.co/models
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
    image_uri = f"763104351884.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.2-optimum0.0.27-neuronx-py310-ubuntu22.04"

    # create Hugging Face Model Class
    huggingface_model = HuggingFaceModel(
        image_uri=image_uri,
        env=hub,
        role=role,
    )

    # deploy model to SageMaker Inference
    predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type="ml.inf2.xlarge",
        container_startup_health_check_timeout=1800,
        volume_size=512,
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

