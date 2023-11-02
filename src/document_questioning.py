import timeit
import argparse
from src.utils import setup_dbqa

def document_questioning():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        type=str,
                        default='Wie heißt der Patient?',
                        help='Enter the query to pass into the LLM')
    args = parser.parse_args()

    # Setup DBQA
    start = timeit.default_timer()
    dbqa = setup_dbqa()
    response = dbqa({'query': args.input})
    end = timeit.default_timer()

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

    print(f"Time to retrieve response: {end - start}")