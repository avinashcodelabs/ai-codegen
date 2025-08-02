from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from argparse import ArgumentParser
from dotenv import load_dotenv
import warnings

load_dotenv()
warnings.filterwarnings("ignore")

parser = ArgumentParser();
parser.add_argument("--language", default="python")
parser.add_argument("--task", default="is prime")
args = parser.parse_args()

llm = ChatOpenAI(
    model="deepseek-coder:6.7b",
    openai_api_base="http://localhost:11434/v1",
    openai_api_key="doesn't matter"  
)

code_prompt = PromptTemplate(
    template="write {language} code to {task}",
    input_variables=["language","task"]
)
test_prompt = PromptTemplate(
    template="write unit test for \n {code}",
    input_variables=["code"]
)

code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key="code"
)

test_chain = LLMChain(
    llm=llm,
    prompt=test_prompt,
    output_key="test"
)

chain = SequentialChain(
    chains=[code_chain,test_chain],
    input_variables=["language","task"],
    output_variables=["code","test"]
)

result = chain({
    "language":args.language,
    "task":args.task
})

print(">>>>> GENERATED CODE")
print(result["code"])
print(">>>>> GENERATED TEST")
print(result["test"])