from math import log
from collections import defaultdict
from search_engine_models.inverted_index import InvertedIndex
from utils import binary_search

class BinaryIndependenceModel(InvertedIndex):
    def __init__(self,known_relevant_docs_cnt = 20,docs_to_be_retrieved = 10,inverted_index = InvertedIndex()):
        super().__init__(inverted_index.doc_folder_path,inverted_index.doc_length_file_path, inverted_index.index_file_path)
        self.known_relevant_docs_cnt = known_relevant_docs_cnt
        self.ttl_doc_cnt = len(self.listdir)
        self.docs_to_be_retrieved = docs_to_be_retrieved

    # Preprocess the query like stemming and removing stop words and punctuation
    def preprocess_query(self,query):
        return self.get_token_cnt(query)

    # Calculate the ct for each term t in query by using the number of relevant docs known, total documents and total documents and relevant documents containing term t
    def calc_doc_scores(self, query_term_freq, known_relevant_docs):
        known_relevant_docs = list(known_relevant_docs)[0:self.known_relevant_docs_cnt]
        term_score = defaultdict(lambda:0.0)
        scores = [[0.0,i] for i in range(len(self.listdir))]
        for term in query_term_freq:
            term_in_known_relevant_doc_cnt = 0
            for doc_id in known_relevant_docs:
                pos = binary_search(doc_id, self.index[term])
                if pos != -1:
                    term_in_known_relevant_doc_cnt+=1
            df = len(self.index[term])
            p = term_in_known_relevant_doc_cnt/self.known_relevant_docs_cnt
            r = (df-term_in_known_relevant_doc_cnt)/(self.ttl_doc_cnt-self.known_relevant_docs_cnt)
            if p == 1.0:
                p = 0.999999
            if r == 1.0:
                r = 0.999999
            if p == 0.0:
                p = 0.000001
            if r == 0.0:
                r = 0.000001
            # print(p,r,term)
            term_score[term] = log((p*(1-r))/((1-p)*r)) #ct

            #Add the ct for terms coming in query and document to RSV of the document
            for docid_tf in self.index[term]:
                doc_id,_ = [int(x) for x in docid_tf.split("_")]
                scores[doc_id][0] += term_score[term]*query_term_freq[term]

        #Sort documents by RSV value in descending order
        scores.sort(reverse=True)
        return scores

        

    # Get the top 10 relevant docs for a query based on the scores
    def get_top_k_relevant_docs(self,scores):
        relevant_docs = []
        for i in range(self.docs_to_be_retrieved):
            relevant_docs.append([scores[i][1],self.listdir[scores[i][1]]])
        return relevant_docs

    # Retrieve the top 10 relevant docs for a query vector
    def return_relevant_docs_for_query(self,query,known_relevant_docs):
        query_term_freq = self.preprocess_query(query)
        scores = self.calc_doc_scores(query_term_freq,known_relevant_docs)
        relevant_docs = self.get_top_k_relevant_docs(scores)
        return relevant_docs 
