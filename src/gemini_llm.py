from typing import Callable
import os
from dotenv import load_dotenv


import pandas as pd


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from .utils import get_categories
from .model import Model


print("Loading environment variables...")
print(f"Enviroment vars loaded: {load_dotenv('../.env')}") 
GEMINI_KEY = os.getenv("GEMINI_ACCESS_TOKEN")



system_prompt_simple_template = """
You are a helpful assistant that helps the user categorize a company name into one of the following categories:
{categories}
If the company name does not fit into any of the categories, please respond with "Other".
Please respond with the category name only, without any additional text.
"""

user_prompt_simple_template = """
Categorize the following company name: {company_name}
"""

system_prompt_info_template = """
You are a helpful assistant that helps the user categorize a company name into one of the following categories:
{categories}
Along the name you will receive a short description of the company. Use this information to categorize the company name into one of the categories.
If the company name does not fit into any of the categories, please respond with "Other".
Please respond with the category name ONLY, without any additional text.
"""

user_prompt_info_template = """
Categorize the following company name: {company_name}
Description: {description}
"""

CATEGORIES = get_categories(pd.read_csv("../resources/categories.csv"))

system_prompt_simple = SystemMessagePromptTemplate.from_template(system_prompt_simple_template,
                                                          partial_variables={"categories": 
                                                                             "\n".join(CATEGORIES)}
                                                          )

system_prompt_info = SystemMessagePromptTemplate.from_template(system_prompt_info_template,
                                                            partial_variables={"categories": 
                                                                                 "\n".join(CATEGORIES)}
                                                            )

user_prompt_simple = HumanMessagePromptTemplate.from_template(user_prompt_simple_template,
                                                            input_variables=["company_name"])


user_prompt_info = HumanMessagePromptTemplate.from_template(user_prompt_info_template,
                                                            input_variables=["company_name", "description"])



chat_prompt_simple = ChatPromptTemplate.from_messages([system_prompt_simple, user_prompt_simple])
chat_prompt_info = ChatPromptTemplate.from_messages([system_prompt_info, user_prompt_info])



class GeminiLLM(Model):

    def __init__(self, model_name: str = "gemini-2.0-flash-lite", 
                 temperature: float = 0.0):
        
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=1000,
            google_api_key=GEMINI_KEY
        )
        self.chat_prompt_simple = chat_prompt_simple
        self.chat_prompt_info = chat_prompt_info

    def categorize(self, company_name: str, description: str = None) -> str:
        if description:
            prompt = self.chat_prompt_info
            input_variables = {"company_name": company_name, "description": description}
        else:
            prompt = self.chat_prompt_simple
            input_variables = {"company_name": company_name}

        chain = prompt | self.llm
        response = chain.invoke(input_variables)
        response = response.content.strip()
        if response not in CATEGORIES:
            print(f"[Warning] Response not in categories: {response}")
            return "Other"
        return response
    
