from config.config import DATA_DIR  
from dotenv import load_dotenv
import os

# Load environment variables
if load_dotenv():
    print(".env file loaded successfully")
else:
    print("Failed to load .env file")

# Print current working directory
print("Current Working Directory:", os.getcwd())

# Access the PYTHONPATH variable
python_path = os.getenv('PYTHONPATH')
print("PYTHONPATH:", python_path)

def main():
    # Call the main functions from each module
    print("Hello World")
    print(python_path)

if __name__ == "__main__":
    main()