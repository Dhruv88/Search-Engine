from csv import writer


"""
Class to calculate evaluation metrics for a query or a set of queries
"""
class EvaluationMetrics:
    def __init__(self, ttl_relevant_cnt=50, ttl_returned=10, query_set_length=1, query_set_metrics=[], file_path="metrics.csv"):
        self.ttl_retrieved = ttl_returned
        self.ttl_relevant_cnt = ttl_relevant_cnt
        self.query_set_length = query_set_length
        self.query_set_metrics = query_set_metrics
        self.file_path = file_path
        self.avg_precision = 0.0
        self.avg_recall = 0.0
        self.map = 0.0
        self.avg_time = 0.0
        self.max_time = 0.0
        self.min_time = 0.0

    # Calculate average precision over all queries
    def set_avg_precision_over_query_set(self, precision):
        self.avg_precision = round(sum(precision)/self.query_set_length, 3)

    # Calculate average recall over all queries
    def set_avg_recall_over_query_set(self, recall):
        self.avg_recall = round(sum(recall)/self.query_set_length, 3)

    # Calculate map over all queries
    def set_MAP_over_query_set(self, ap):
        self.map = round(sum(ap)/self.query_set_length, 3)

    # Calculate average running time over all queries
    def set_avg_running_time_per_query(self, time):
        self.avg_time = round(sum(time)/self.query_set_length, 3)
        self.max_time = round(max(time),3)
        self.min_time = round(min(time),3)

    # Calculate all average metrics over all queries
    def get_avg_metrics_over_query_set(self):
        precision = []
        recall = []
        ap = []
        time = []
        for i in range(self.query_set_length):
            precision.append(self.query_set_metrics[i][1])
            recall.append(self.query_set_metrics[i][2])
            ap.append(self.query_set_metrics[i][3])
            time.append(self.query_set_metrics[i][4])

        self.set_avg_precision_over_query_set(precision)
        self.set_avg_recall_over_query_set(recall)
        self.set_MAP_over_query_set(ap)
        self.set_avg_running_time_per_query(time)

    # Calculate precision for a single query given the number of relevant docs retrieved
    def precision(self,actual_relevant_docs_cnt):
        return round(actual_relevant_docs_cnt/self.ttl_retrieved,3)

    # Calculate recall for a single query given the number of relevant docs retrieved
    def recall(self,actual_relevant_docs_cnt):
        return round(actual_relevant_docs_cnt/self.ttl_relevant_cnt,3)        

    # Calculate ap for a single query given the ranks of relevant docs retrieved
    def AP(self,actual_relevant_docs_pos):
        avg_precision = 0.0
        for i in range(len(actual_relevant_docs_pos)):
            avg_precision += ((i+1)/(actual_relevant_docs_pos[i]+1))
        if len(actual_relevant_docs_pos) > 0:
            avg_precision/=len(actual_relevant_docs_pos)
        return round(avg_precision,3)

    # Calculate all metrics for a single query given number and ranks of actual relevant docs retrieved
    def evaluate_metrics(self,actual_relevant_docs_cnt, actual_relevant_docs_pos):
        precision = self.precision(actual_relevant_docs_cnt)
        recall = self.recall(actual_relevant_docs_cnt)
        ap = self.AP(actual_relevant_docs_pos)
        return precision,recall,ap

    # Store all metrics to a csv file
    def store_metrics_to_csv(self):
         with open(self.file_path, 'w', newline='', encoding="utf8") as fp:
            writ = writer(fp)
            writ.writerow(["query_id","precision","recall","ap","running time"])
            for query_metric in self.query_set_metrics:
                writ.writerow(query_metric)
            writ.writerow(["avg_precision", self.avg_precision])
            writ.writerow(["avg_recall", self.avg_recall])
            writ.writerow(["map", self.map])
            writ.writerow(["avg_running_time_per_query", self.avg_time])
            writ.writerow(["max_running_time_per_query", self.max_time])
            writ.writerow(["min_running_time_per_query", self.min_time])