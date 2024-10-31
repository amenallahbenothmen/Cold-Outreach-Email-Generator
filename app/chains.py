import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:

    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-70b-versatile")


    
    def extract_jobs(self, cleaned_text):
        prompt = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from a career page of a website and describes a single job posting.
            Extract and return the details in JSON format, including the following keys:
            
            - `title`: The main position title for the job.
            - `requirements`: A list of all specific requirements mentioned for the job.
            - `responsibilities`: A concise summary of the main responsibilities for this role.
            - `experience_years`: The required experience in years as an integer.
            - If specific years are provided, use that value.
            - If experience is implied (e.g., "senior" or "junior"), set `experience_years` to `>0`.
            - If no indication of experience is given, set it to 0.

            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):    
            """
        )
        chain_extract =prompt | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]  


    def extact_readme(self,text):

        prompt = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM GITHUB README:
            {page_data}
            
            ### INSTRUCTION:
            The scraped text above is from a GitHub README file that provides details about an individual's professional background. 
            Your task is to extract and format the following information as valid JSON:
            
            - `role`: A list of roles or titles of the individual, with the main or primary role listed first. If no specific role is mentioned, use the area the individual is specializing in as the primary role.
            - `experience_duration`: The number of years of experience as an integer. If the individual is a student, set `experience_duration` to `0`.
            - `experience`: A brief summary of relevant experience, including areas of expertise, specific domains, or specializations (e.g., data science, machine learning, cloud computing).
            - `skills`: A list of key technical and non-technical skills highlighted in the README, such as programming languages, frameworks, tools, and soft skills.
            - `description`: A concise summary that introduces the individual's background, education, or current focus.
            
            Please follow these rules:
            - Only include information that is explicitly mentioned in the README text.
            - For `experience_duration`, provide an integer (e.g., `2` for two years of experience, or `0` if the individual is a student).
            - If no specific role is found, infer the primary role based on the individual's area of specialization.
            - If a specific field is not available, return an empty string ("") for text fields or an empty list ([]) for the `skills` field.
            - Do not add any explanatory text outside of the JSON format.
            
            ### VALID JSON (NO PREAMBLE) 

            """
        )
        chain_extract= prompt | self.llm 
        res = chain_extract.invoke(input={'page_data':text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res] 

