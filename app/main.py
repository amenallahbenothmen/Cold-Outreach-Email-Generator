import streamlit as st
from chain import process_job_posting, process_user_profile, generate_application_letter
from utils import load_environment_variables

# Set page configuration
st.set_page_config(
    page_title="Indeed Job Application Letter Generator",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="expanded",
)

def main():
    st.title("Indeed Job Application Letter Generator")

    st.markdown("""
    Welcome to the Indeed Job Application Letter Generator! This tool helps you create a personalized job application letter by analyzing a job posting and your GitHub profile.
    """)

    st.sidebar.header("Configuration")
    st.sidebar.markdown("Enter the required information below:")

    github_username = st.sidebar.text_input("GitHub Username", value="", help="Enter your GitHub username.")
    job_posting_url = st.sidebar.text_input("Job Posting URL", value="", help="Enter the URL of the Indeed job posting.")

    generate_button = st.sidebar.button("Generate Application Letter")

    if generate_button:
        if not github_username or not job_posting_url:
            st.error("Please provide both GitHub username and job posting URL.")
            return

        st.info("Loading environment variables...")
        try:
            env_vars = load_environment_variables()
        except Exception as e:
            st.error(f"Error loading environment variables: {e}")
            return

        st.info("Processing job posting...")
        job_offer = process_job_posting(job_posting_url, env_vars)

        st.info("Processing your GitHub profile...")
        user_data = process_user_profile(github_username, env_vars)

        st.info("Generating application letter...")
        application_letter = generate_application_letter(job_offer, user_data, env_vars)

        st.success("Application letter generated successfully!")

        st.header("Your Personalized Application Letter")
        st.write(application_letter)

        # Optionally, provide a download button
        st.download_button(
            label="Download Letter as Text File",
            data=application_letter,
            file_name="application_letter.txt",
            mime="text/plain",
        )

if __name__ == "__main__":
    main()
