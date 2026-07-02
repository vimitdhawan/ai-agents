import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

client = InferenceClient(
    model="moonshotai/Kimi-K2.5",
    token=os.getenv("HF_TOKEN"),
)


def observation(messages):
    response = client.chat.completions.create(
        messages=messages,
        stop=["Observation:"],
        max_tokens=2048,
        extra_body={'thinking': {'type': 'disabled'}},
        stream=False,
    )
    return response.choices[0].message.content


def chat(messages):
    response = client.chat.completions.create(
        messages=messages,
        max_tokens=1024,
        extra_body={'thinking': {'type': 'disabled'}},
        stream=False,
    )
    return response.choices[0].message.content