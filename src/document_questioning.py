import timeit
import argparse
import box
import yaml
from src.utils import setup_dbqa, setup_dbqa_fact_checking

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

def document_questioning() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        type=str,
                        default='Wie heißt der Patient?',
                        help='Enter the query to pass into the LLM')
    args = parser.parse_args()

    # Setup DBQA
    startQA = timeit.default_timer()
    dbqa = setup_dbqa()
    response = dbqa({'query': args.input})
    endQA = timeit.default_timer()

    print(f'\nAnswer: {response["result"]}')
    print('='*50)

    # Process source documents
    source_docs = response['source_documents']
    for i, doc in enumerate(source_docs):
        print(f'\nSource Document {i+1}\n')
        print(f'Source Text: {doc.page_content}')
        print(f'Document Name: {doc.metadata["source"]}')
        print(f'Page Number: {doc.metadata.get("page", 1)}\n')
        print('='* 60)

    print(f"Time to retrieve response: {endQA - startQA}")

    if cfg.FACTCHECKING == True:
        startFactCheck = timeit.default_timer()
        dbqafact = setup_dbqa_fact_checking()
        response_fact = dbqafact({'query': response["result"]})
        endFactCheck = timeit.default_timer()
        print("Factcheck:")
        print(f'\nAnswer: {response_fact["result"]}')
        print('='*50)

        # Process source documents
        source_docs = response_fact['source_documents']
        for i, doc in enumerate(source_docs):
            print(f'\nSource Document {i+1}\n')
            print(f'Source Text: {doc.page_content}')
            print(f'Document Name: {doc.metadata["source"]}')
            print(f'Page Number: {doc.metadata.get("page", 1)}\n')
            print('='* 60)

        print(f"Time to retrieve fact check: {endFactCheck - startFactCheck}")