import sys
from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex
import gradio as gr
import os
import logging

api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def create_index(dir_path):
    documents = SimpleDirectoryReader(dir_path).load_data()
    index = GPTSimpleVectorIndex.from_documents(documents)

    try:
        index.save_to_disk('index.json')
        print("Index saved to disk")
    except Exception as e:
        print(f"Error saving index to disk: {e}")


def chat(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="tree_summarize")
    return response.response


interface = gr.Interface(fn=chat,
                         inputs=gr.components.Textbox(lines=5, label="Ask a question about Levi"),
                         outputs="text",
                         title="ChatGPT Powered Resume Bot"
                         )

index = create_index("docs")
interface.launch(share=True)