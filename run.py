from search_engine_models.binary_independence_model import BinaryIndependenceModel
from evaluation_metrics import EvaluationMetrics
from search_engine_models.language_model import LanguageModel
from search_engine_models.elastic import Elastic
from utils import load_all_queries, load_relevant_docs_for_each_query, stem_content
from search_engine_models.vector_model import VectorModel
from search_engine_models.inverted_index import InvertedIndex

from query import Query
import timeit
##########################Construct Inverted Index###############################
# Create inverted index for the docs
index = InvertedIndex()
index.process_all_documents()

##########################Vector Model###############################
# Create vector model load the index
vector_model = VectorModel()
vector_model.load_index_from_csv()
vector_model.load_doc_lengths_from_csv()
print("loading index done")

# load all queries and relevant docs for them
relevant_docs_per_query = load_relevant_docs_for_each_query(vector_model.listdir)
queries = load_all_queries()
print("loading queries done")

#without pseudo relevance feedback
print("Running Queries")
overall_metrics = EvaluationMetrics(query_set_length=len(queries), query_set_metrics=[], file_path="metrics\metrics_tfidf.csv")
for query_id in queries.keys():
    query = Query(query_id, queries[query_id], relevant_docs_per_query[query_id])
    t = timeit.repeat(lambda: vector_model.return_relevant_docs_for_query(query.query), number=3, repeat=3)
    time = round(sum(t)/len(t),3)
    retrieved_docs = vector_model.return_relevant_docs_for_query(query.query)
    query.cnt_actual_relevant_docs_returned_and_ranks(retrieved_docs)
    precision, recall, ap = query.evaluate_metrics(query.actual_relevant_docs_cnt, query.actual_relevant_docs_pos)
    overall_metrics.query_set_metrics.append([query_id, precision, recall, ap, time])

overall_metrics.get_avg_metrics_over_query_set()

print("Storing Metrics")
overall_metrics.store_metrics_to_csv()

print("Done!")

##########################Pseudo Relevance Feedback###############################
# with pseudo relevance feedback and get best alpha
print("Running Queries to get best alpha that maximises map")

alpha = 0.0
max_map = 0.0
best_alpha = 0.0
while alpha <= 1:
    overall_metrics = EvaluationMetrics(query_set_length=len(queries), query_set_metrics=[],file_path="metrics\metrics_with_feedback.csv")
    for query_id in queries.keys():
        query = Query(query_id, queries[query_id], relevant_docs_per_query[query_id])
        # t = timeit.repeat(lambda: vector_model.return_relevant_docs_for_query_with_feedback(query.query, alpha=0.1), number=3, repeat=3)
        # time = round(sum(t)/len(t),3)
        
        retrieved_docs = vector_model.return_relevant_docs_for_query_with_feedback(query.query, alpha=alpha)
        query.cnt_actual_relevant_docs_returned_and_ranks(retrieved_docs)
        precision, recall, ap = query.evaluate_metrics(query.actual_relevant_docs_cnt, query.actual_relevant_docs_pos)
        overall_metrics.query_set_metrics.append([query_id, precision, recall, ap, 0])
        # print(precision,recall,ap)

    overall_metrics.get_avg_metrics_over_query_set()
    if max_map < overall_metrics.map:
        max_map = overall_metrics.map
        best_alpha = alpha
    print(alpha,overall_metrics.map,max_map)
    alpha += 0.1
    

print(best_alpha,"gives max map as", max_map)
print("Done!")

#########################Language Model###############################
language_model = LanguageModel()
language_model.load_index_from_csv()
print("loading index done")

language_model.make_language_model()
print("Made language model")

# load all queries and relevant docs for them
relevant_docs_per_query = load_relevant_docs_for_each_query(language_model.listdir)
queries = load_all_queries()
print("loading queries done")

print("Running Queries")
overall_metrics = EvaluationMetrics(query_set_length=len(queries), query_set_metrics=[],file_path="metrics\metrics_lm.csv")
for query_id in queries.keys():
    query = Query(query_id, queries[query_id], relevant_docs_per_query[query_id])
    t = timeit.repeat(lambda: language_model.return_relevant_docs_for_query(query.query), number=3, repeat=3)
    time = round(sum(t)/len(t),3)
    retrieved_docs = language_model.return_relevant_docs_for_query(query.query)
    # print(retrieved_docs)
    query.cnt_actual_relevant_docs_returned_and_ranks(retrieved_docs)
    precision, recall, ap = query.evaluate_metrics(query.actual_relevant_docs_cnt, query.actual_relevant_docs_pos)
    # print(precision,recall,ap,time)
    overall_metrics.query_set_metrics.append([query_id, precision, recall, ap, time])

overall_metrics.get_avg_metrics_over_query_set()

print("Storing Metrics")
overall_metrics.store_metrics_to_csv()

print("Done!")


#########################Binary Independence Model###############################
binary_independence_model = BinaryIndependenceModel(known_relevant_docs_cnt=20)
binary_independence_model.load_index_from_csv()
print("loading index done")

# load all queries and relevant docs for them
relevant_docs_per_query = load_relevant_docs_for_each_query(binary_independence_model.listdir)
queries = load_all_queries()
print("loading queries done")


print("Running Queries")
overall_metrics = EvaluationMetrics(query_set_length=len(queries), query_set_metrics=[], file_path="metrics\metrics_bim.csv")
for query_id in queries.keys():
    query = Query(query_id, queries[query_id], relevant_docs_per_query[query_id])
    t = timeit.repeat(lambda: binary_independence_model.return_relevant_docs_for_query(query.query,query.relevant_docs), number=1, repeat=1)
    time = round(sum(t)/len(t),3)
    retrieved_docs = binary_independence_model.return_relevant_docs_for_query(query.query,query.relevant_docs)
    query.cnt_actual_relevant_docs_returned_and_ranks(retrieved_docs)
    precision, recall, ap = query.evaluate_metrics(query.actual_relevant_docs_cnt, query.actual_relevant_docs_pos)
    overall_metrics.query_set_metrics.append([query_id, precision, recall, ap, time])
    # print(precision,recall,ap,time)
    # break

overall_metrics.get_avg_metrics_over_query_set()

print("Storing Metrics")
overall_metrics.store_metrics_to_csv()

print("Done!")

#########################ElasticSearch###############################
vector_model = VectorModel()
elastic = Elastic()
# load all queries and relevant docs for them
relevant_docs_per_query = load_relevant_docs_for_each_query(vector_model.listdir)
queries = load_all_queries()
print("loading queries done")

print("Running Queries")
overall_metrics = EvaluationMetrics(query_set_length=len(queries), query_set_metrics=[],file_path="metrics\metrics_es.csv")
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