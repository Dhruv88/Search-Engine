from evaluation_metrics import EvaluationMetrics


class Query(EvaluationMetrics):
    def __init__(self, query_id, query, relevant_docs=[]):
        self.query_id = query_id
        self.query = query
        self.relevant_docs = relevant_docs
        super().__init__(len(relevant_docs))
    
    #ttl actual relevant docs retrieved and their positions
    def cnt_actual_relevant_docs_returned_and_ranks(self,docs_retrieved):
        self.ttl_retrieved = len(docs_retrieved)
        relevant_docs_pos = []
        for i in range(len(docs_retrieved)):
            if docs_retrieved[i][0] in self.relevant_docs:
                relevant_docs_pos.append(i)
        
        self.actual_relevant_docs_cnt, self.actual_relevant_docs_pos = len(relevant_docs_pos), relevant_docs_pos
    