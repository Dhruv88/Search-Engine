from collections import defaultdict
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from ignore_lists import *

# Read all content of a text file
def read_text_file(file_path):
    with open(file_path, 'r', encoding="utf8") as f:
        return f.read()

# Stem the content of doc/query and also remove stop words and punctuation 
def stem_content(content):
    words = word_tokenize(content)
    # words = re.split(' |\n|/', content)
    tokens = []
    ps = PorterStemmer()
    for w in words:
        w = w.strip("\"^._|\\")
        if (w!="" and w[0] not in ignore and w not in stop_words) or w.isnumeric():
            w = w.replace('-', '/')
            w = w.replace('.', '/')
            w = w.replace('_', '/')
            w = w.replace('^', '/')
            w = w.replace('=', '/')
            w = w.replace(':', '/')
            w = w.replace('\"', '/')
            w = w.replace('\'', '/')
            w = w.replace(',', '/')
            spl = w.split('/')
            if(len(spl)>1):
                for ws in spl:
                    if (ws!="" and ws[0] not in ignore and ws not in stop_words) or ws.isnumeric():
                        ws = ws.strip("\",.^_|\\")    
                        rootWord=ps.stem(ws)
                        tokens.append(rootWord)
            else:
                rootWord=ps.stem(w)
                tokens.append(rootWord)
    
    return tokens

# Load query id as key and relevant doc set as value from output.txt
def load_relevant_docs_for_each_query(doc_list,file_path="./Docs/output.txt"):
    relevant_docs_per_query = defaultdict()
    with open(file_path, 'r', encoding="utf8") as f:
        while True:
            line = f.readline()
            if line == "":
                break
            query_id,rel_doc_name = line.split(" ")[0:2]
            query_id = int(query_id)
            if query_id not in relevant_docs_per_query.keys():
                relevant_docs_per_query[query_id] = set()
            if rel_doc_name in doc_list:
                relevant_docs_per_query[query_id].add(doc_list.index(rel_doc_name))

    return relevant_docs_per_query

# Load all queries from query.txt
def load_all_queries(file_path="./Docs/query.txt"):
    queries = defaultdict()
    with open(file_path, 'r', encoding="utf8") as f:
        while True:
            line = f.readline()
            if line == "":
                break
            if line == "\n":
                continue
            query_id,query = line.split("  ")
            query_id = int(query_id)
            queries[query_id]=query

    return queries

def binary_search(doc_id, posting_list):
    l = 0
    r = len(posting_list)-1
    while l<=r:
        mid = (l+r)//2
        cur_doc_id,_ = posting_list[mid].split("_")
        cur_doc_id = int(cur_doc_id)
        if doc_id == cur_doc_id:
            return mid
        elif doc_id < cur_doc_id:
            r = mid-1
        else:
            l = mid+1

    return -1