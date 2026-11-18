import pandas as pd
import os
import sys
import gdown
#from getpass import getpass
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# ==========================================
#  PART 1: AUTOMATIC FILE DOWNLOADER
# ==========================================

files_to_download = {
    "Datasets/saas_docs.csv":         "https://drive.google.com/file/d/1RElOhN7bYsDAJUNQhYyqM7IzX-Xo6myq/view?usp=sharing",
    "Datasets/credit_card_terms.csv": "https://drive.google.com/file/d/1_giivc_B0urOKpct0XY2yVZuxW3Eenuf/view?usp=sharing",
    "Datasets/hospital_policy.csv":   "https://drive.google.com/file/d/1pL7OnDhnmz9pteIpBJ12gu2_ixrc2hPm/view?usp=sharing",
    "Datasets/ecommerce_faqs.csv":    "https://drive.google.com/file/d/1O4fTjsLFbz55oOiwJUwLwZryO5OSSF6p/view?usp=sharing"
}

print("--- Downloading Files from Google Drive ---")
for filename, url in files_to_download.items():
    if not os.path.exists(filename):
        gdown.download(url, filename, quiet=False)
        print(f"Downloaded: {filename}")
    else:
        print(f"Skipped: {filename} (Already exists)")
print("--- Download Complete ---\n")


# ==========================================
#  PART 2: AI AGENT SETUP (MULTI-FILE)
# ==========================================

# 1. SETUP: Get API Key Securely
print("ENTER YOUR OPENAI API KEY BELOW:")
#api_key = getpass()
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 2. LOAD ALL CSVs INTO A LIST
dataframes = [] # We will store all the loaded tables here
loaded_names = []

try:
    for filename in files_to_download.keys():
        df = pd.read_csv(filename)
        dataframes.append(df)
        loaded_names.append(filename)
        print(f"SUCCESS: Loaded '{filename}' ({len(df)} rows)")

except Exception as e:
    print(f"\nERROR loading files: {e}")
    sys.exit()

# 3. DEFINE THE RULES
system_prompt = """
You are a smart data assistant capable of reading multiple CSV files.
- You have access to 4 different datasets: SaaS Docs, Credit Card Terms, Hospital Policy, and Ecommerce FAQs.
- When asked a question, determine which DataFrame is most relevant.
- Do NOT answer from general knowledge.
- Answer in plain English.
"""

try:
    # ---------------------------------------------------------
    # TODO 1: Initialize the LLM
    # Hint: Use ChatOpenAI with model="gpt-4o-mini" and temperature=0.0
    # ---------------------------------------------------------
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0,api_key=api_key)

    # ---------------------------------------------------------
    # TODO 2: Create the Pandas Agent
    # Hint: Pass the 'llm' and the list 'dataframes' as arguments.
    # Set verbose=True and allow_dangerous_code=True
    # ---------------------------------------------------------
    agent = create_pandas_dataframe_agent(
        llm,
        dataframes,
        verbose=True,
        agent_type="openai-functions",
        allow_dangerous_code=True
    )

    print("\nAI Agent is ready! You can ask questions across ALL files.")
    print("Example: 'What is the visiting hour in the hospital?' or 'What is the API limit?'")

except Exception as e:
    print(f"Error initializing agent: {e}")
    sys.exit()

while True:
    try:
        user_input = input("\nEnter your question (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Exiting the program. Goodbye!")
            break

        # ---------------------------------------------------------
        # TODO 3: Get the agent's response
        # Hint: Use agent.run(user_input) to get the answer.
        # ---------------------------------------------------------
        response = agent.run(user_input)
        print(f"\nAgent Response:\n{response}")

    except Exception as e:
        print(f"Error during interaction: {e}")