from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from utils import get_github_readme_raw_link, get_user_projects, register_data_in_chromadb, query_for_application_letter
import chromadb

def process_job_posting(job_posting_url, env_vars):
    # Load the job posting content
    loader = WebBaseLoader(job_posting_url)
    page_data = loader.load().pop().page_content

    # Define the prompt for extracting job details
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

    # Initialize the language model client with the API key
    llm = ChatGroq(
        temperature=0, 
        groq_api_key=env_vars['GROQ_API_KEY'], 
        model_name="llama-3.1-70b-versatile"
    )

    # Generate the response
    chain_extract = prompt | llm 
    res = chain_extract.invoke(input={'page_data': page_data})

    # Parse the JSON output
    json_parser = JsonOutputParser()
    job_offer = json_parser.parse(res.content)
    return job_offer

def process_user_profile(github_username, env_vars):
    # Process GitHub README
    readme_url = get_github_readme_raw_link(github_username)
    loader = WebBaseLoader(readme_url)
    page_data = loader.load().pop().page_content

    # Define the prompt for extracting user details
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

    # Initialize the language model client
    llm = ChatGroq(
        temperature=0, 
        groq_api_key=env_vars['GROQ_API_KEY'], 
        model_name="llama-3.1-70b-versatile"
    )

    # Generate the response
    chain_extract = prompt | llm 
    res = chain_extract.invoke(input={'page_data': page_data})

    # Parse the JSON output
    json_parser = JsonOutputParser()
    readme_data = json_parser.parse(res.content)

    # Process GitHub projects
    projects = get_user_projects(github_username, env_vars['GITHUB_TOKEN'], llm)

    # Register data in ChromaDB
    client = chromadb.Client()
    collection = client.create_collection(name='user_data_collection')

    register_data_in_chromadb(collection, readme_data, projects)

    user_data = {
        'readme': readme_data,
        'projects': projects,
        'collection': collection
    }
    return user_data

def generate_application_letter(job_offer, user_data, env_vars):
    job_title = job_offer.get("title", "AI-related role")
    job_requirements = "; ".join(job_offer.get("requirements", []))
    job_responsibilities = job_offer.get("responsibilities", "N/A")

    # Query the database for relevant projects
    query_results = query_for_application_letter(job_title, job_requirements, job_responsibilities, user_data['collection'])

    # Initialize the language model client
    llm = ChatGroq(
        temperature=0, 
        groq_api_key=env_vars['GROQ_API_KEY'], 
        model_name="llama-3.1-70b-versatile"
    )

    # Generate the application letter
    prompt_template = PromptTemplate.from_template(
        """
        ### JOB AND USER INFORMATION:
        Job Title: {job_title}
        Requirements: {job_requirements}
        Responsibilities: {job_responsibilities}
        Relevant User Projects:
        {relevant_projects}

        ### INSTRUCTION:
        Write a job application letter for the user, emphasizing their relevant skills and experiences in alignment with the job title, requirements, and responsibilities. Mention specific projects by name, placing the project name in parentheses, and briefly explain how each project supports the user’s qualifications and demonstrates their expertise in relation to the job’s demands.

        ### APPLICATION LETTER:
        """
    )

    # Format the relevant projects
    relevant_projects = "\n".join(
        [f"- {meta.get('project_name', 'N/A')}: {doc}" for doc, meta in zip(query_results['documents'][0], query_results['metadatas'][0])]
    )

    # Generate the letter
    chain = prompt_template | llm
    result = chain.invoke(input={
        'job_title': job_title,
        'job_requirements': job_requirements,
        'job_responsibilities': job_responsibilities,
        'relevant_projects': relevant_projects
    })

    return result.content.strip()
