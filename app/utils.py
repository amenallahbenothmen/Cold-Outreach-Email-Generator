import os
from dotenv import load_dotenv, find_dotenv
import requests
from langchain_core.prompts import PromptTemplate

def load_environment_variables():

    dotenv_path = find_dotenv()
    if not dotenv_path:
        raise FileNotFoundError("Could not find .env file.")
    load_dotenv(dotenv_path)

    github_token = os.getenv("GITHUB_TOKEN")
    groq_api_key = os.getenv("GROQ_API_KEY")


    if not github_token:
        raise ValueError("GITHUB_TOKEN is not set in the environment variables.")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is not set in the environment variables.")

    env_vars = {
        'GITHUB_TOKEN': github_token,
        'GROQ_API_KEY': groq_api_key
    }


    return env_vars

def get_github_readme_raw_link(username):
    return f"https://raw.githubusercontent.com/{username}/{username}/main/README.md"

def get_project_content(username, repo_name, token):
    headers = {"Authorization": f"token {token}"}
    contents_url = f"https://api.github.com/repos/{username}/{repo_name}/contents"


    requirements_content = None
    readme_content = None


    requirements_response = requests.get(f"{contents_url}/requirements.txt", headers=headers)
    if requirements_response.status_code == 200:
        download_url = requirements_response.json().get("download_url")
        if download_url:
            requirements_content = requests.get(download_url).text


    readme_response = requests.get(f"{contents_url}/README.md", headers=headers)
    if readme_response.status_code == 200:
        download_url = readme_response.json().get("download_url")
        if download_url:
            readme_content = requests.get(download_url).text

    return requirements_content, readme_content

def llm_summarize(content, llm):
    prompt_template = PromptTemplate.from_template(
        """
        ### CONTENT TO SUMMARIZE:
        {content}

        ### INSTRUCTION:
        Provide a concise summary of the content above, focusing on the main technologies, functionality, and purpose if relevant.

        ### SUMMARY:
        """
    )
    chain = prompt_template | llm
    result = chain.invoke(input={'content': content})
    return result.content.strip()

def generate_llm_project_summary(repo_name, language, requirements_content, readme_content, llm):
    summarized_requirements = llm_summarize(requirements_content, llm) if requirements_content else ""
    summarized_readme = llm_summarize(readme_content, llm) if readme_content else ""

    input_data = (
        f"Project Name: {repo_name}\n"
        f"Primary Language: {language}\n"
        f"Requirements Summary: {summarized_requirements}\n"
        f"README Summary:\n{summarized_readme}"
    )

    prompt_template = PromptTemplate.from_template(
        """
        ### PROJECT INFORMATION:
        {input_data}

        ### INSTRUCTION:
        Based on the project name, language, requirements, and README summary above, provide:
        - Project Type (e.g., Machine Learning, Data Visualization, Web Application)
        - Main Technologies used in the project
        - The primary goal or purpose of the project

        ### OUTPUT (AS THREE SEPARATE LINES WITHOUT LABELS):
        """
    )

    llm_chain = prompt_template | llm
    result = llm_chain.invoke(input={'input_data': input_data})

    output_lines = result.content.strip().splitlines()
    project_type = output_lines[0] if len(output_lines) > 0 else ""
    technologies = output_lines[1] if len(output_lines) > 1 else ""
    primary_goal = output_lines[2] if len(output_lines) > 2 else ""

    return project_type, technologies, primary_goal

def get_user_projects(username, token, llm):
    headers = {"Authorization": f"token {token}"}
    projects = []
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        repos = response.json()
        for repo in repos[1:]:  # Skip the first repository if it's the profile README
            repo_name = repo['name']
            repo_url = repo['html_url']
            language = repo['language']
            requirements_content, readme_content = get_project_content(username, repo_name, token)
            project_type, technologies, primary_goal = generate_llm_project_summary(repo_name, language, requirements_content, readme_content, llm)
            project_data = {
                "Project Name": repo_name,
                "URL": repo_url,
                "Primary Language": language,
                "Project Type": project_type,
                "Main Technologies": technologies.split(", "),
                "Primary Goal": primary_goal
            }
            projects.append(project_data)
    else:
        print(f"Failed to fetch repositories: {response.status_code}")
    return projects

def register_data_in_chromadb(collection, readme_data, projects):

    role_str = ", ".join(readme_data['role']) if isinstance(readme_data['role'], list) else readme_data['role']
    skills_str = ", ".join(readme_data['skills']) if isinstance(readme_data['skills'], list) else readme_data['skills']
    collection.add(
        documents=[readme_data['description']],
        ids=["readme"],
        metadatas={
            "type": "README",
            "role": role_str,
            "experience_duration": readme_data['experience_duration'],
            "experience": readme_data['experience'],
            "skills": skills_str
        }
    )

    for project in projects:
        main_technologies_str = ", ".join(project["Main Technologies"]) if isinstance(project["Main Technologies"], list) else project["Main Technologies"]
        collection.add(
            documents=[project["Primary Goal"]],
            ids=[project["Project Name"]],
            metadatas={
                "type": "Project",
                "project_name": project["Project Name"],
                "url": project["URL"],
                "primary_language": project["Primary Language"],
                "project_type": project["Project Type"],
                "main_technologies": main_technologies_str
            }
        )

def query_for_application_letter(job_title, job_requirements, job_responsibilities, collection):
    query_text = (
        f"Generate a job application letter for the position '{job_title}'. "
        f"Focus on user projects, skills, and experiences that demonstrate alignment with the following requirements:\n{job_requirements}\n"
        f"and responsibilities:\n{job_responsibilities}\n"
        "Identify relevant information to emphasize the user's qualifications and enthusiasm for the position."
    )

    results = collection.query(
        query_texts=[query_text],
        n_results=5
    )
    return results
