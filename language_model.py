from collections import defaultdict
from inverted_index import InvertedIndex
from utils import stem_content

class LanguageModel(InvertedIndex):
    def __init__(self,inverted_index=InvertedIndex(), docs_to_be_retrieved = 10, lmbda = 0.5):
        self.docs_to_be_retrieved = docs_to_be_retrieved
        self.collection_size = 0
        self.general_model = defaultdict(lambda:0.0)
        super().__init__(inverted_index.doc_folder_path,inverted_index.doc_length_file_path, inverted_index.index_file_path)
        self.doc_size = [0]*len(self.listdir)
        self.document_models = []
        for i in range(len(self.listdir)):
            self.document_models.append(defaultdict(lambda:0.0))
        self.lmbda = lmbda

    def cnt_term_freqs(self):
        for term in self.index.keys():
            ttl_term_freq = 0
            for docid_tf in self.index[term]:
                doc_id, tf = [int(x) for x in docid_tf.split('_')]
                self.document_models[doc_id][term] = tf
                if self.document_models[5701]["oil"] == 4:
                    print(doc_id, tf)
                self.doc_size[doc_id] += tf
                ttl_term_freq += tf
            self.general_model[term] = ttl_term_freq
            self.collection_size += ttl_term_freq

    def make_general_model(self):
        for term in self.general_model.keys():
            self.general_model[term] /= self.collection_size

    def make_doc_models(self):
        for doc_id in range(len(self.document_models)):
            for term in self.document_models[doc_id].keys():
                if self.doc_size[doc_id] == 0:
                    self.document_models[doc_id][term] = 0.0
                else:
                    self.document_models[doc_id][term] /= self.doc_size[doc_id]

    def make_language_model(self):
        self.cnt_term_freqs()
        self.make_general_model()
        self.make_doc_models()

    def preprocess_query(self,query):
        return self.get_token_cnt(query)
        
    def calc_term_score(self,term,i):
        score = self.lmbda*self.document_models[i][term] + (1-self.lmbda)*self.general_model[term]
        if score == 0.0:
            score = 0.0001
        return score
        
    def calc_doc_scores(self, query_term_freq):
        scores = [[1000.0,i] for i in range(len(self.listdir))]
        for i in range(len(self.document_models)):
            for term in query_term_freq:
                term_score = self.calc_term_score(term,i)
                scores[i][0] *= query_term_freq[term]*term_score
        
        scores.sort(reverse=True)

        return scores


    def get_top_k_relevant_docs(self,scores):
        relevant_docs = []
        for i in range(self.docs_to_be_retrieved):
            relevant_docs.append([scores[i][1],self.listdir[scores[i][1]]])
        return relevant_docs
        
    def return_relevant_docs_for_query(self,query):
        query_term_freq = self.preprocess_query(query)
        scores = self.calc_doc_scores(query_term_freq)
        relevant_docs = self.get_top_k_relevant_docs(scores)
        return relevant_docs
        