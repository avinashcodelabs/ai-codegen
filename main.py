from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from argparse import ArgumentParser
from dotenv import load_dotenv

load_dotenv()

parser = ArgumentParser();
parser.add_argument("--language", default="python")
parser.add_argument("--task", default="is prime")
args = parser.parse_args()

llm = OpenAI() # This will look for OPENAI_API_KEY in env variables by default.

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

