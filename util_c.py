
import torch
from transformers import pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from pymongo import MongoClient
import os


def get_embeddings_with_model_name(model_name):
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'}, encode_kwargs={'device': 'cpu'})

@staticmethod
def get_embeddings():
    # model_name = "intfloat/multilingual-e5-large"
    model_name = "./multilingual-e5-large"
    embeddings = get_embeddings_with_model_name(model_name)
    return embeddings


# -----------------------------------------------------------------------
embeddings = None
if not embeddings:
    embeddings = get_embeddings()
# -----------------------------------------------------------------------

@staticmethod
def get_pipe():
    # model_name = "HuggingFaceH4/zephyr-7b-beta"
    # model_name = "HuggingFaceH4/zephyr-7b-gemma-v0.1"
    # model_name = "/content/zephyr-7b-gemma-v0.1-coupang"
    # model_name = "/content/zephyr-7b-gemma-v0.1-kr"
    # model_name = "/content/zephyr-7B-Beta-Chat"
    model_name = "/content/zephyr-7b-gemma-v0.1"
    pipe = get_pipe_with_model_name(model_name)
    return pipe


def get_pipe_with_model_name(model_name):
    pipe = pipeline(
        "text-generation",
        model=model_name,
        device_map="cuda",
        torch_dtype=torch.bfloat16
    )
    return pipe


# -----------------------------------------------------------------------
pipe = None
if not pipe:
    pipe = get_pipe()
# -----------------------------------------------------------------------

def get_database(db_name):
    MONGO_URI = "mongodb+srv://ysjeong:jeong7066#@cluster0.jf3wpr7.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

    # Connect to MongoDB
    client = MongoClient(MONGO_URI)
    return client[db_name]

def get_db_table(db_name, table_name):
    mdb_database = get_database(db_name)
    db_table = mdb_database[table_name]
    return db_table

def get_tb_counsel():
    DB_NAME = "db_counsel"
    TB_COUNSEL = "tb_counsel"
    return get_db_table(DB_NAME, TB_COUNSEL)


# get file name
def get_file_name(f_path):
    split_char = "/"
    if "/" not in f_path:
        split_char = "\\"

    f_name = f_path.split(split_char)[-1].replace(".pdf", "").strip()
    return f_name.strip()


"""
  make_prompt
"""
def make_prompt(query, query_list_txt, inquiry_examples_txt):
    prompt = f"""
          You are a query maker bot. Your task is to choose only one query below
          and choose most match query after <<< >>> into one of the following predefined query list:

          ####
          query list:

{query_list_txt}
          ####


          If the Inquiry doesn't fit into any of the above query list, classify it as:
          not matched

          You will only respond with the predefined query list.
          Don't provide additional explanations or text.
          You must reply with one of the query list without any change.
          Don't add additional comment or text.


          ####
          Here are some Inquiry / query examples:

{inquiry_examples_txt}
          ####

          <<<
        Inquiry: {query}
          >>>

    """
    return prompt


"""
  make_prompt
"""
def make_prompt_of_qas_list(query, qas_list):

    qas_text_arr = []
    for idx, qa in enumerate(qas_list):
        q = qas_list[idx]["question"]
        a = qas_list[idx]["answer"]
        idx += 1
        # qas_text_arr.append(f"{idx}.Question: {q}\n A: {a}\n")
        qas_text_arr.append(f"{idx}.Question: \n{q}\n")
        qas_text_arr.append(f"{idx}.Answer: \n{a}\n\n")
    qas_text = "".join(qas_text_arr)

    prompt = f"""
          You are a Inquiry answer bot. You will be given some Question and Answer sets.
          Your task is to generate the appropriate reply for Inquery based on Question and Answer sets.
          Inquery after <<< >>> :

          ####
          Question and Answer sets:

{qas_text}
          ####

          You must make a reply very related to Question and Answer sets.
          You should reference Question and Answer sets to generate the reply to Inquiry.
          The reply text content should be within the Answer of Question and Answer sets.
          Do not say that you can not reply.
          Do not refer about Question and Answer sets itself including the No of Question and Answer sets.
          You must always reply in Korean.
          Your answer should be logical and make sense.

          <<<
        Inquiry: {query}
          >>>

    """

    return prompt


"""
  get_answer_by_llm 실행
  기능 : qas_list (Q/A list)를 참조하여 query에 적합한 답변을 생성한다.
"""
def get_answer_by_llm(query, qas_list):
    prompt = make_prompt_of_qas_list(query, qas_list)
    messages = [
        {
            "role": "system",
            "content": "",  # Model not yet trained for follow this
        },
        {"role": "user", "content": prompt},
    ]
    max_new_tokens = 1024
    outputs = pipe(
        messages,
        # max_new_tokens=128,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        stop_sequence="<|im_end|>",
    )
    print(outputs[0]["generated_text"][-1]["content"])

    answer = outputs[0]["generated_text"][-1]["content"]

    return answer


"""
  get_answer_by_embedding
"""
def get_answer_by_embedding(embeddings, tb_counsel, query):

    query_embedding = embeddings.embed_documents([query.strip()])[0]

    # Retrieve relevant child documents based on query
    child_qas = tb_counsel.aggregate([
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding_q",
                "queryVector": query_embedding,
                "numCandidates": 10,
                "limit": 1
            }
        },
        {
            "$project": {
                "question": 1,
                "answer": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ])

    child_qas_list = list(child_qas)

    answer = ""
    if len(child_qas_list) > 0:
        print(child_qas_list[0]["question"])
        print("========================================")
        print(child_qas_list[0]["answer"])
        print("========================================")
        print(child_qas_list[0]["score"])

    question = child_qas_list[0]["question"]
    answer = child_qas_list[0]["answer"]
    score = float(child_qas_list[0]["score"])

    return question, answer, score, child_qas_list


def querying(query, history):

    tb_counsel = get_tb_counsel()

    answer = ""
    score = -1
    llm_answer = ""

    process_type = "Embedding"
    question, answer, score, qas_list = get_answer_by_embedding(embeddings, tb_counsel, query)

    # if score < 0.97:
    #     llm_answer = get_answer_by_llm(query, qas_list)
    llm_answer = get_answer_by_llm(query, qas_list)

    return_text_arr = []
    return_text_arr.append(f"<h2>Process type</h2>\n{process_type}")
    return_text_arr.append(f"<h2>Question</h2>\n{question}")
    return_text_arr.append(f"<h2>Answer</h2>\n{answer}")
    if process_type == "Embedding":
        return_text_arr.append(f"<h2>Score</h2>\n{score}")
    if len(llm_answer) > 0:
        return_text_arr.append(f"<h2>LLM answer</h2>\n{llm_answer}")
    return_text = "".join(return_text_arr)

    return return_text