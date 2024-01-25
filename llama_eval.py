from langchain.retrievers.web_research import WebResearchRetriever

import os
from langchain.vectorstores import Chroma
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.chat_models.openai import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.chains import RetrievalQAWithSourcesChain
import torch

import re
from typing import List
from langchain.chains import LLMChain
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.output_parsers.pydantic import PydanticOutputParser

from langchain.llms import LlamaCpp
from langchain.embeddings import GPT4AllEmbeddings
from langchain.embeddings import LlamaCppEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import faiss
from langchain.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import HuggingFaceEmbeddings

from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain

from torch import cuda, bfloat16
import transformers
from langchain.schema import Document
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_content_from_url(soup):
    """Get the text from the soup

    Args:
        soup (BeautifulSoup): The soup to get the text from

    Returns:
        str: The text from the soup
    """
    text = ""
    tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5']
    for element in soup.find_all(tags):  # Find all the <p> elements
        text += element.text + "\n"
    return text


def perform_google_search(api_key, cx, query, num_pages=1, results_per_page=5):
    all_results = []
    search_results = []
    for page in range(num_pages):
        start_index = page * results_per_page + 1

        url = f'https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={cx}&start={start_index}&num={results_per_page}'

        response = requests.get(url)

        if response.status_code == 200:

            data = response.json()

            items = data.get('items', [])

            if items:
                all_results.extend(items)
            else:
                print(f"No results on page {page + 1}.")
                break
        else:
            print(f"Error: {response.status_code}\n{response.text}")
            break

    if all_results:
#        print("\nSearch Results:")
        for idx, item in enumerate(all_results, start=1):
            title = item.get('title', 'N/A')
            link = item.get('link', 'N/A')
            snippet = item.get('snippet', 'N/A')
           # print(f"{idx}. Title: {title}\n   Link: {link}\n   Snippet: {snippet}\n")

            if "youtube.com" in link or "youtube.in" in link or "instagram.com" in link:
                continue
            search_results.append({'title': title, 'link': link, 'snippet': snippet})
            # search_results.append(link)
    else:
        print("No search results found.")

    return search_results

def pipelineQA(query):
    web_links = perform_google_search(api_key, cx, query)
    documents = createDocs(web_links)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    all_splits = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    flag = 0
    vectorstore = []
    # storing embeddings in the vector store
    if flag == 0:
#        print("Create vectorstore")
        vectorstore = FAISS.from_documents(all_splits, embeddings)
        flag = 1
    else:
#        print("Adding docs to vectorstore")
        vectorstore.add_documents(all_splits)

    chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True, combine_docs_chain_kwargs={"prompt": prompt})
    # Run the chain to get the result
    chat_history = []
    result = chain({"question": user_input, "chat_history": chat_history})
    #print(result['source_documents'])
    return result


def createDocs(web_links):
    documents = []

    for page in web_links:
        html_doc = requests.get(page['link'])
        soup = BeautifulSoup(html_doc.text, 'html.parser')
#        print("New soup\n")
        if soup:
            # print(get_content_from_url(soup))
            text = get_content_from_url(soup)

            metadata = {
                'title': page['title'],
                'link': page['link'],
                'snippet': page['snippet']
            }

            document = Document(text=text, metadata=metadata, page_content=html_doc.text)

            documents.append(document)
    return documents

access_token = "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"
model_id = 'meta-llama/Llama-2-7b-chat-hf'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# begin initializing HF items, you need an access token
hf_auth = '<add your access token here>'
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    token=access_token
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    token=access_token
)

# enable evaluation mode to allow model inference
model.eval()

print(f"Model loaded on {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    token=access_token
)

from transformers import pipeline

pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_new_tokens=1024,
                do_sample=True,
                top_k=30,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
                )

# LLM
llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature': 0, 'streaming': True})

# print(llm(prompt="Why is the sky blue?"))

# Search
# os.environ["GOOGLE_CSE_ID"] = "a474d79b089b44319"
# os.environ["GOOGLE_API_KEY"] = "AIzaSyCk3mNY0ZPoStwdFdCg5HFKzHS-NuVClyg"
# search = GoogleSearchAPIWrapper()
#
#
#
# # vectorstore
# embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# embedding_size = 768
# index = faiss.IndexFlatL2(embedding_size)
# vectorstore = FAISS(embedding_function.embed_query, index, InMemoryDocstore({}), {})
#
#
# web_research_retriever_llm_chain = WebResearchRetriever.from_llm(
#     vectorstore=vectorstore,
#     llm=llm,
#     search=search,
#     num_search_results=4
# )
# user_input = input("\nPlease enter a question: ")
# Run
# docs = web_research_retriever_llm_chain.get_relevant_documents(user_input)
# print("\n Docs: \n")
# print(docs)
# print("\n")
# print(len(docs))

# print("\n")

# from langchain.text_splitter import RecursiveCharacterTextSplitter

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
# all_splits = text_splitter.split_documents(docs)

# storing embeddings in the vector store
# vectorstore = FAISS.from_documents(all_splits, embedding_function)

from langchain.chains import ConversationalRetrievalChain

prompt_template = """Use the following pieces of context and if required your own knowledge to generate a concise and informative and accurate
 answer to the question at the end. And don't ask any follow up questions and don't write any unhelpful answer and unnecessary comments.
 Also don't make up any answers.
If the context has references from an online forum like quora or reddit prefer most upvoted answers.
{context}

Question: {question}
Helpful Answer:"""

prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True,combine_docs_chain_kwargs={"prompt": prompt})

# result = chain({"question": user_input,"chat_history": chat_history})
# print(result)
# print("\n")

# print(result['answer'])
# print(result['source_documents'])
# Loop through the list of source documents

# print("Souces:\n")
# for doc in result['source_documents']:
# Print the metadata source of each document
# print(doc.metadata['source'])


from langchain.evaluation.qa import QAEvalChain

import json

no_query = 50
with open('output.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    first_5 = data[:no_query]
    data = first_5

# print(first_5)

#chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True,combine_docs_chain_kwargs={"prompt": prompt})

results = []
chat_history = []

prompt_template_eval = """Question: {question}
Answer: {answer}
Result: {result}

Grade:

Please grade the result based on the question and the answer (a answer is any of the comma separated answers)
on a scale of 0 to 5, where 0 means the result is completely wrong or irrelevant, and 5 means the result is completely correct and comprehensive.

Consider the following criteria when grading the result:

- Accuracy: Does the answer match the facts and the logic of the question and the real_answer?
- Completeness: Does the answer cover all the aspects and details of the question and the real_answer?
- Clarity: Is the answer easy to understand and well-written?
- Relevance: Is the answer related to the question and not off-topic or redundant?

Grade:
"""

from langchain.prompts.prompt import PromptTemplate

_PROMPT_TEMPLATE = """
You are given a question, the student's answer, and a list of acceptable variations of the true answer,
 and are asked to grade the student answer as either CORRECT or INCORRECT. Don't add any unnecessary spaces at the end.

Example Format:
QUESTION: question here
STUDENT ANSWER: student's answer here
TRUE ANSWER: list of acceptable variations of the true answer here
GRADE: CORRECT or INCORRECT here

Grade the student answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. 
It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements. Begin! 

QUESTION: {query}
STUDENT ANSWER: {result}
TRUE ANSWER: {answer}
GRADE:
"""

template = """
Comment on the similarity between the answer_two and answer_one. If they are similar, comment TRUE else FALSE. Don't respond with anything else.

answer_one: {answer}
answer_two: {result}
comment:

"""

PROMPT = PromptTemplate(input_variables=["query", "answer", "result"], template=_PROMPT_TEMPLATE)
#api_key = 'AIzaSyAsTjJHuQG6_aKV4TQZLHdGIw1jyiHIlFQ'
cx = '2762757efa3e6403f'
api_key = 'AIzaSyCk3mNY0ZPoStwdFdCg5HFKzHS-NuVClyg'
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}

count = 0
for item in data:
    print("Starting iteration ", count)

    user_input = item['query']
    answer = item['answer']
    if count == 14:
      results.append({'query': user_input, 'answer': answer, 'result': 'why did chicken cross the road'})
      count += 1
      continue
    if count == 34:
      results.append({'query': user_input, 'answer': answer, 'result': 'To inventor Albert Einstein'})
      count += 1
      continue
    result = pipelineQA(user_input)
    # result = chain({"question": user_input, "chat_history": chat_history})

    results.append({'query': user_input, 'answer': answer, 'result': result['answer']})
    chat_history = []
    count += 1

    # results.append((user_input, result['answer'], result['source_documents']))
    # chat_history = []

print(results)

import os
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "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" #key
#os.environ["OPENAI_API_KEY"] = "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"
llm_model = "gpt-3.5-turbo"
gpt = ChatOpenAI(temperature=0, model=llm_model)

# predictions = chain.apply(first_5)


print("\nCalling QAEvalChain")
eval_chain = QAEvalChain.from_llm(gpt, prompt=PROMPT)  # prompt_template=prompt_template_eval)
print("\nCalling eval_chain")
graded_outputs = eval_chain.evaluate(data, results, question_key="query", answer_key="answer", prediction_key="result")
#graded_outputs = eval_chain.evaluate(data, results, answer_key="answer", prediction_key="result")
print("\nGraded output:")
#for graded_output in graded_outputs:
#    print(graded_output)

count_correct = 0
count_incorrect = 0
count_extra = 0
pairs = []

for i, eg in enumerate(data):
   # print(graded_outputs[i]['results'])
    if 'INCORRECT' in graded_outputs[i]['results']:
        count_incorrect += 1
    elif 'CORRECT' in graded_outputs[i]['results']:
        count_correct += 1
    else:
        count_extra += 1
    pairs.append([results[i]['query'], results[i]['answer'], results[i]['result'], graded_outputs[i]['results']])

print("\nPairs: ")
for pair in pairs:
    print("Query: ", pair[0])
    print("Real answer: ", pair[1])
    print("Predicted Answer: ", pair[2])
    print("Predicted Grade: ", pair[3])
    print("\n")

print("Total queries: ", no_query)
print("No corrects: ", count_correct)
print("No incorrects: ", count_incorrect)
print("No extras: ", count_extra)
print("Accuracy: ", (count_correct / no_query) * 100)


# graded_outputs[0]