
import torch,json,requests,transformers
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFacePipeline
from langchain import PromptTemplate
from torch import cuda, bfloat16
from langchain.schema import Document 
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from transformers import pipeline


def get_content_from_url(soup):
    """Get the text from the soup

    Args:
        soup (BeautifulSoup): The soup to get the text from

    Returns:
        str: The text from the soup
    """
    text = ""
    tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5']
    for element in soup.find_all(tags):  
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



### IMPORTANT CONFIG


#Hugging face access token
access_token = "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"

#LAMMA Model ID
model_id = 'meta-llama/Llama-2-7b-chat-hf'


#GOOGLE API KEY AND TOKEN
cx = '2762757efa3e6403f'
api_key = 'AIzaSyCk3mNY0ZPoStwdFdCg5HFKzHS-NuVClyg'

#embedding model
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}




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



device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'





#Reading questions from file, generating answers through pipeline and store in NaturalQuestionsAnswer.json file
with open('naturalQuestion.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

count = 0
for item in data:
    print("Starting iteration ", count)
    
    user_input = item['question']
    answer = item['answer']
    print(f"question is {user_input}")
    result = pipelineQA(user_input)
    dictionary = {"query": user_input,"answer":result['answer']}
    json_object = json.dumps(dictionary, indent=4)
    with open("NaturalQuestionsAnswer.json", "a") as outfile:
        outfile.write(json_object)
        outfile.write(",\n")
    count+=1
    
    """
    NOTE After entire json is completed append remove "," from last line in NaturalQuestionsAnswer.json file and enclose everything in [] brackets to make proper JSON format 
    For eg:
   [
    {
    "query": "What is the best way to learn a new language?",
    "answer": " The best way to learn a new language depends on each learner and their specific goals. Explore media at various stages to develop your sense of the language, practice with apps, podcasts, and other programs, and learn from mistakes. Consistency is key for undertaking any new task, and learning a new language requires a reliable schedule to form the habit.\n\nAdditional Information:\n\n* Research indicates that humans benefit from forming good habits.\n* Stay consistent with your learning goals and set aside time to study every day or every few days.\n* Language-learning programs can provide structured lessons and feedback from native speakers.\n* Immersion in the local culture can also help learners develop their language skills.\n* Free foreign language podcasts and apps can be a useful resource for learners.\n* Don't practice in isolation; get feedback from native speakers to improve your skills.\n* Don't worry about making mistakes, as they are an essential part of the learning process."
    },
    {
        "query": "Who is the funniest comedian of all time?",
        "answer": " It is difficult to determine who the funniest comedian of all time is, as comedy is subjective and what one person finds hilarious, another might not find as funny. However, some comedians who are widely considered to be among the funniest of all time include Dave Chappelle, Jerry Seinfeld, and Richard Pryor. These comedians have been widely praised for their unique styles and ability to make audiences laugh.\n\nNote: The answer provided is based on the information available on the internet and may not be a comprehensive or definitive list of the funniest comedians of all time."
    },

    Need to be manually modified as

       {
    "query": "What is the best way to learn a new language?",
    "answer": " The best way to learn a new language depends on each learner and their specific goals. Explore media at various stages to develop your sense of the language, practice with apps, podcasts, and other programs, and learn from mistakes. Consistency is key for undertaking any new task, and learning a new language requires a reliable schedule to form the habit.\n\nAdditional Information:\n\n* Research indicates that humans benefit from forming good habits.\n* Stay consistent with your learning goals and set aside time to study every day or every few days.\n* Language-learning programs can provide structured lessons and feedback from native speakers.\n* Immersion in the local culture can also help learners develop their language skills.\n* Free foreign language podcasts and apps can be a useful resource for learners.\n* Don't practice in isolation; get feedback from native speakers to improve your skills.\n* Don't worry about making mistakes, as they are an essential part of the learning process."
    },
    {
        "query": "Who is the funniest comedian of all time?",
        "answer": " It is difficult to determine who the funniest comedian of all time is, as comedy is subjective and what one person finds hilarious, another might not find as funny. However, some comedians who are widely considered to be among the funniest of all time include Dave Chappelle, Jerry Seinfeld, and Richard Pryor. These comedians have been widely praised for their unique styles and ability to make audiences laugh.\n\nNote: The answer provided is based on the information available on the internet and may not be a comprehensive or definitive list of the funniest comedians of all time."
    }
    ]


    
    """