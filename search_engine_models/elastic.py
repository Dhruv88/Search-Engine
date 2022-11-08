from datetime import datetime
from elasticsearch import Elasticsearch
from utils import read_text_file, stem_content
import os

class Elastic():
    es = Elasticsearch(['https://elastic:2vwXocD5-av1_QQs*X-V@localhost:9200'])

    # Retrieve the top 10 relevant docs for a query
    def return_relevant_docs_for_query(self,query):
        json_query = {
            "match": {
                "text": query
            }
        }
        res = self.es.search(index="q3",query=json_query)
        docs = []
        for num, doc in enumerate(res['hits']['hits']):
            cur = []
            cur.append(int(doc['_source']['docID']))
            cur.append(doc['_source']['docName'])
            docs.append(cur)
        return docs


"""
Class to ingest docs to ElasticSearch
"""
class ElasticPut():
    es = Elasticsearch(['https://elastic:2vwXocD5-av1_QQs*X-V@localhost:9200'])

    def __init__(self, doc_folder_path="./alldocs"):
        self.doc_folder_path = doc_folder_path
        os.chdir(doc_folder_path)
        self.listdir = os.listdir()
        os.chdir("..")


    # Process all docs to create the inverted index
    def process_all_documents(self):
        print('here')
        cnt = 0
        doc_term = []
        for i in range(len(self.listdir)):
        # Check whether file is in text format or not
            # if file.endswith(".txt"):
            file_path = f"{self.doc_folder_path}\{self.listdir[i]}"
            # call read text file function
            content = read_text_file(file_path)
            tokens = stem_content(content)
            content = ""
            for token in tokens:
                content+=' '+token
            # print(content)
            doc = {
                'docID': str(i),
                'docName': self.listdir[i],
                'text': content,
            }
            res = self.es.index(index="q3", id=i+1, document=doc)
            print(res['result'])
            cnt+=1
            if(cnt % 100==0):
                # pprint(index)
                print(str(cnt)+" file done")
        print('testing for query: describe history oil industry')
        
    
    def test(self):
        query= {
                "bool": {
                    "should": {
                        "match": {      
                            "text": "pearl farming operations actual farming operations described culturing pearls japanese pearl productions status pearl farming production"
                        }
                    }
                }
            }
        res = self.es.search(index="q3", query=query)
        print("Got %d Hits:" % res['hits']['total']['value'])
        for hit in res['hits']['hits']:
            print("%(docName)s %(docID)s" % hit["_source"])
            


# feed = ElasticPut()
# feed.process_all_documents()
# feed.test()
# queryClass = Elastic()
# tokens = stem_content("describe history oil industry")
# content = ""
# for token in tokens:
#     content+=' '+token
# docs = queryClass.return_relevant_docs_for_query(content)
# print(docs)