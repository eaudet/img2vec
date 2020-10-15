import sys
import os
import json
import time
sys.path.append("../img2vec_pytorch")  # Adds higher directory to python modules path.
from img2vec_pytorch.img_to_vec import Img2Vec
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from elasticsearch import RequestsHttpConnection


def index_batch(docs):
    requests = []
    for i, doc in enumerate(docs):
        request = doc
        img = Image.open(os.path.join(input_path, doc["image"]))
        vec = img2vec.get_vec(img)
        request["_op_type"] = "index"
        request["_index"] = INDEX_NAME
        request["image_vector"] = vec
        requests.append(request)
    bulk(client, requests)


def index_data():
    print("Creating the 'posts' index.")
    client.indices.delete(index=INDEX_NAME, ignore=[404])

    with open(INDEX_FILE) as index_file:
        source = index_file.read().strip()
        client.indices.create(index=INDEX_NAME, body=source)

    docs = []
    count = 0

    with open(DATA_FILE) as data_file:
        for line in data_file:
            line = line.strip()

            doc = json.loads(line)
            # if doc["type"] != "question":
            #     continue

            docs.append(doc)
            count += 1

            if count % BATCH_SIZE == 0:
                index_batch(docs)
                docs = []
                print("Indexed {} documents.".format(count))

        if docs:
            index_batch(docs)
            print("Indexed {} documents.".format(count))

    client.indices.refresh(index=INDEX_NAME)
    print("Done indexing.")

def run_query_loop():
    while True:
        try:
            handle_query()
        except KeyboardInterrupt:
            return

def handle_query():
    query = input("Enter file name: ")

    embedding_start = time.time()
    img = Image.open(os.path.join(input_path, query))
    query_vector = img2vec.get_vec(img)
    embedding_time = time.time() - embedding_start

    script_query = {
        "knn": {
            "image_vector": {
                "vector": query_vector,
                "k": 2
            }
        }
    }

    search_start = time.time()
    response = client.search(
        index=INDEX_NAME,
        body={
            "size": SEARCH_SIZE,
            "query": script_query,
            "_source": {"includes": ["title"]}
        }
    )
    search_time = time.time() - search_start

    print()
    print("{} total hits.".format(response["hits"]["total"]["value"]))
    print("embedding time: {:.2f} ms".format(embedding_time * 1000))
    print("search time: {:.2f} ms".format(search_time * 1000))
    for hit in response["hits"]["hits"]:
        print("id: {}, score: {}".format(hit["_id"], hit["_score"]))
        print(hit["_source"])
        print()

if __name__ == '__main__':
    INDEX_NAME = "images"

    INDEX_FILE = "indexImage.json"

    DATA_FILE = "images.json"

    BATCH_SIZE = 1000

    SEARCH_SIZE = 5

    GPU_LIMIT = 0.5


    input_path = './test_images'

    img2vec = Img2Vec()

    client = Elasticsearch(connection_class=RequestsHttpConnection, http_auth=('admin', 'admin'), use_ssl=True, verify_certs=False)

    index_data()

    run_query_loop()

