import time
from dotenv import load_dotenv

import gradio as gr
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from openai import OpenAI

# Read environment variables from .env
load_dotenv()

# --- OpenAI Python client test ---
client = OpenAI()

response = client.chat.completions.create(
    model='gpt-5-mini',
    messages=[
        {"role": "system", "content": "You are an AI assistant"},
        {"role": "user", "content": "Hi, how are you?"},
    ],
    max_completion_tokens=100
)

reply = response.choices[0].message.content

print(f'OpenAI client: {reply}')


# --- LangChain ChatOpenAI test ---
client = ChatOpenAI(
    model="gpt-5-mini",
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant"),
    ("human", "{message}"),
])

chain = prompt | client | StrOutputParser()
response = chain.invoke({"message": "Hi, how are you?"})

print(f'LangChain ChatOpenAI: {response}')


# --- Gradio UI test ---
def fake_gan():
    time.sleep(1)
    images = [
            "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=387&q=80",
            "https://images.unsplash.com/photo-1554151228-14d9def656e4?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=386&q=80",
            "https://images.unsplash.com/photo-1542909168-82c3e7fdca5c?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MXx8aHVtYW4lMjBmYWNlfGVufDB8fDB8fA%3D%3D&w=1000&q=80",
    ]
    return images

demo = gr.Interface(
    fn=fake_gan,
    inputs=None,
    outputs=gr.Gallery(label="Generated Images", columns=2),
    title="FD-GAN",
    description="This is a fake demo of a GAN. In reality, the images are randomly chosen from Unsplash.",
    api_name="predict",
)

demo.launch(share=True)