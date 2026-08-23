from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from state import ResearchState
from tools import tools

load_dotenv()

llm = ChatGroq(
    model_name = os.environ['GROQ_MODEL']
)

def route_node(query: str, state: ResearchState) -> dict:
    
    prompt = ChatPromptTemplate.from_template(
        """"""
    )