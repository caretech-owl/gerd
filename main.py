import box
import yaml
from dotenv import find_dotenv, load_dotenv
from src.document_generation import document_generation
from src.document_questioning import document_questioning

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

if __name__ == "__main__":
    if cfg.DOCUMENT_GENERATION==True:
        document_generation()
    else:
        document_questioning()
        