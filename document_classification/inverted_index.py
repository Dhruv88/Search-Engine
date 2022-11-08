from collections import Counter, OrderedDict, defaultdict
from math import sqrt,log
from csv import writer, reader
from utils import read_text_file, stem_content
import os

"""
Class to build inverted index given a set of documents
"""
class InvertedIndex:
    def __init__(self, doc_folder_path="./alldocs", doc_length_file_path="doc_length_clf.csv", index_file_path="inverted_index_clf.csv"):
        self.doc_folder_path = doc_folder_path
        self.doc_length_file_path = doc_length_file_path
        self.index_file_path = index_file_path
        # ordered dictionary with defaulty factory as empty list
        self.index = defaultdict(list)
        self.doc_lengths = defaultdict(float)
        os.chdir(doc_folder_path)
        self.listdir = os.listdir()
        os.chdir("..")

    # Calculate idf of a term
    def idf(self, term):
        if term in self.index.keys():
            if len(self.index[term])==0:
                return log(len(self.listdir)) - log(1)
            return log(len(self.listdir)) - log(len(self.index[term]))
        else: # Assuming that the term occurs in one of the documents
            return log(len(self.listdir)) - log(1)

    # Process all docs to create the inverted index
    def process_all_documents(self,ignore=set()):
        cnt = 0
        doc_term = [Counter() for i in range(len(self.listdir))]
        for i in range(len(self.listdir)):
        # Check whether file is in text format or not
            # if file.endswith(".txt"):
            if self.listdir[i] not in ignore:
                file_path = f"{self.doc_folder_path}\{self.listdir[i]}"
                # call read text file function
                content = read_text_file(file_path)
                tokens_freq_cnt = self.get_token_cnt(content)
                doc_term[i] = tokens_freq_cnt
                self.update_index(i, tokens_freq_cnt)
                cnt+=1
                if(cnt % 100==0):
                    # pprint(index)
                    print(str(cnt)+" file done")
        self.calc_doc_length_and_store(doc_term)
        self.store_index_to_csv()
    
    # Parse a document and get all terms in it with their frequency
    def get_token_cnt(self,content):
        tokens = stem_content(content)
        return Counter(tokens)


    # Calculate the doc vector lengths and store them in csv file
    def calc_doc_length_and_store(self,doc_term):
        with open(self.doc_length_file_path, 'w', newline='', encoding="utf8") as fp:
            for i in range(len(doc_term)):
                doc_terms_freq = doc_term[i]
                length = 0
                for term in doc_terms_freq:
                    tf = doc_terms_freq[term]
                    idf = self.idf(term)
                    score = tf*idf
                    length += (score**2)
                length = sqrt(length)
            
                writ = writer(fp)
                writ.writerow([str(i), str(length)])


    # Store the index to csv file
    def store_index_to_csv(self):
        with open(self.index_file_path, 'w', newline='', encoding="utf8") as fp:
            writ = writer(fp)
            for token in self.index:
                row = [token] + self.index[token]
                # print(row)
                writ.writerow(row)


    # Given the terms in a doc and their frequency update the index with it
    def update_index(self,doc_id, tokens_freq_cnt):
        for token in tokens_freq_cnt:
            entry = str(doc_id)+'_'+str(tokens_freq_cnt[token]) 
            if token not in self.index.keys():
                self.index[token] = []
            self.index[token].append(entry)

    # Load an already created index from csv
    def load_index_from_csv(self):
        with open(self.index_file_path, 'r', encoding="utf8") as fp:
            for row in reader(fp):
                self.index[row[0]] = row[1:]

    # Load doc vector lengths from csv
    def load_doc_lengths_from_csv(self):
        with open(self.doc_length_file_path, 'r', encoding="utf8") as fp:
            for row in reader(fp):
                self.doc_lengths[int(row[0])] = float(row[1])
