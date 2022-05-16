from inverted_index import InvertedIndex
from collections import defaultdict
from utils import binary_search

"""
Class to implement vector model and tf-idf scoring
"""
class VectorModel(InvertedIndex):
    def __init__(self, inverted_index = InvertedIndex(),docs_to_be_retrieved = 10):
        self.docs_to_be_retrieved = docs_to_be_retrieved
        super().__init__(inverted_index.doc_folder_path,inverted_index.doc_length_file_path, inverted_index.index_file_path)

    # Preprocess the query like stemming and removing stop words and punctuation
    def preprocess_query(self,query):
        return self.get_token_cnt(query)

    # Calculate the score for all the docs w.r.t. a query and sort them in descending order
    def calc_doc_scores(self, query_vector,ignore):
        scores = [[0.0,i] for i in range(len(self.listdir))]
        for term in query_vector.keys():
            idf = self.idf(term)
            query_score = query_vector[term]
            for docid_tf in self.index[term]:
                doc_id,tf_doc = docid_tf.split("_")
                doc_id = int(doc_id)
                tf_doc = int(tf_doc)
                scores[doc_id][0] += tf_doc*idf*query_score

        scores.sort(reverse=True)
        # counter=0
        for i in range(len(scores)):
            if scores[i][0] == 0 and i>10:
                break
            # print(scores[i][0], scores[i][1])
            if scores[i][0]  in ignore:
                continue
            if self.doc_lengths[scores[i][1]] == 0:
                # counter += 1
                print(self.listdir[scores[i][1]])
            else:
                scores[i][0] /= self.doc_lengths[scores[i][1]]

        scores.sort(reverse=True)
        # print(counter)

        return scores
    # Get the top k relevant docs for a query based on the scores
    def get_top_k_relevant_docs(self,scores):
        relevant_docs = []
        for i in range(self.docs_to_be_retrieved):
            relevant_docs.append([scores[i][1],self.listdir[scores[i][1]]])
        return relevant_docs
    
    def set_docs_to_be_retrieved(self,k):
        self.docs_to_be_retrieved = k


    
    # Retrieve the top 10 relevant docs for a query vector
    def return_relevant_docs_for_query(self,query):
        query_vector = self.make_query_vector(query)
        return self.return_relevant_docs_for_query_vector(query_vector) 

    def return_top_k_nearest_docs(self,doc,ignore): #modfied
        doc_vector = self.make_query_vector(doc)
        return self.return_relevant_docs_for_query_vector(doc_vector,ignore)

    def return_relevant_docs_for_query_vector(self,query_vector,ignore):
        scores = self.calc_doc_scores(query_vector,ignore)
        relevant_docs = self.get_top_k_relevant_docs(scores)
        return relevant_docs 

    def calc_centroid(self,relevant_docs):  
        centroid = defaultdict(lambda: 0.0)
        for term in self.index.keys():
            postinglist = self.index[term]
            idf = self.idf(term)
            for relevant_doc in relevant_docs:
                doc_id,_ = relevant_doc
                pos = binary_search(doc_id,postinglist)
                if pos != -1:
                    _,tf = postinglist[pos].split("_")
                    tf = int(tf)
                    centroid[term] = centroid[term] + tf*idf
        return centroid  


    def make_query_vector(self,query):  
        query_term_freq = self.preprocess_query(query)
        # print(query_term_freq)
        query_vector = defaultdict(lambda: 0.0)
        for term in query_term_freq:
            query_vector[term] = query_term_freq[term]*self.idf(term) 
        return query_vector  


