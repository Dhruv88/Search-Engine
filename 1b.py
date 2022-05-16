from utils import stem_content
from evaluation_metrics import EvaluationMetrics
from utils import load_all_queries, load_relevant_docs_for_each_query
from query import Query
from elastic import Elastic
from vector_model import VectorModel
import timeit

vector_model = VectorModel()
elastic = Elastic()
# load all queries and relevant docs for them
relevant_docs_per_query = load_relevant_docs_for_each_query(vector_model.listdir)
queries = load_all_queries()
print("loading queries done")

print("Running Queries")
overall_metrics = EvaluationMetrics(query_set_length=len(queries), query_set_metrics=[],file_path="metrics_1b.csv")
for query_id in queries.keys():
    query = Query(query_id, queries[query_id], relevant_docs_per_query[query_id])
    tokens = stem_content(query.query)
    final_query =" "
    for token in tokens:
        final_query += " " + token
    t = timeit.repeat(lambda: elastic.return_relevant_docs_for_query(final_query), number=3, repeat=3)
    time = round(sum(t)/len(t),3)
    retrieved_docs = elastic.return_relevant_docs_for_query(final_query)
    query.cnt_actual_relevant_docs_returned_and_ranks(retrieved_docs)
    precision, recall, ap = query.evaluate_metrics(query.actual_relevant_docs_cnt, query.actual_relevant_docs_pos)
    overall_metrics.query_set_metrics.append([query_id, precision, recall, ap, time])

overall_metrics.get_avg_metrics_over_query_set()

print("Storing Metrics to metrics.csv")
overall_metrics.store_metrics_to_csv()

print("Done!")

