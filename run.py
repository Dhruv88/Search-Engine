from binary_independence_model import BinaryIndependenceModel
from evaluation_metrics import EvaluationMetrics
from language_model import LanguageModel
from utils import load_all_queries, load_relevant_docs_for_each_query
from vector_model import VectorModel
from inverted_index import InvertedIndex

from query import Query
import timeit

##########################Part-1a###############################
# Create inverted index for the docs
index = InvertedIndex()
index.process_all_documents()

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
overall_metrics = EvaluationMetrics(query_set_length=len(queries), query_set_metrics=[], file_path="metrics_tfidf.csv")
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

##########################Part-2###############################
# with pseudo relevance feedback and get best alpha
# print("Running Queries to get best alpha that maximises map")

# alpha = 0.0
# max_map = 0.0
# best_alpha = 0.0
# while alpha <= 1:
#     overall_metrics = EvaluationMetrics(query_set_length=len(queries), query_set_metrics=[],file_path="metrics_with_feedback.csv")
#     for query_id in queries.keys():
#         query = Query(query_id, queries[query_id], relevant_docs_per_query[query_id])
#         # t = timeit.repeat(lambda: vector_model.return_relevant_docs_for_query_with_feedback(query.query, alpha=0.1), number=3, repeat=3)
#         # time = round(sum(t)/len(t),3)
        
#         retrieved_docs = vector_model.return_relevant_docs_for_query_with_feedback(query.query, alpha=alpha)
#         query.cnt_actual_relevant_docs_returned_and_ranks(retrieved_docs)
#         precision, recall, ap = query.evaluate_metrics(query.actual_relevant_docs_cnt, query.actual_relevant_docs_pos)
#         overall_metrics.query_set_metrics.append([query_id, precision, recall, ap, 0])
#         # print(precision,recall,ap)

#     overall_metrics.get_avg_metrics_over_query_set()
#     if max_map < overall_metrics.map:
#         max_map = overall_metrics.map
#         best_alpha = alpha
#     print(alpha,overall_metrics.map,max_map)
#     alpha += 0.1
    

# print(best_alpha,"gives max map as", max_map)
# print("Done!")

##########################Part-3###############################
# Language Model
# language_model = LanguageModel()
# language_model.load_index_from_csv()
# print("loading index done")

# language_model.make_language_model()
# print("Made language model")

# # load all queries and relevant docs for them
# relevant_docs_per_query = load_relevant_docs_for_each_query(language_model.listdir)
# queries = load_all_queries()
# print("loading queries done")

# print("Running Queries")
# overall_metrics = EvaluationMetrics(query_set_length=len(queries), query_set_metrics=[],file_path="metrics_lm.csv")
# for query_id in queries.keys():
#     query = Query(query_id, queries[query_id], relevant_docs_per_query[query_id])
#     t = timeit.repeat(lambda: language_model.return_relevant_docs_for_query(query.query), number=3, repeat=3)
#     time = round(sum(t)/len(t),3)
#     retrieved_docs = language_model.return_relevant_docs_for_query(query.query)
#     # print(retrieved_docs)
#     query.cnt_actual_relevant_docs_returned_and_ranks(retrieved_docs)
#     precision, recall, ap = query.evaluate_metrics(query.actual_relevant_docs_cnt, query.actual_relevant_docs_pos)
#     # print(precision,recall,ap,time)
#     overall_metrics.query_set_metrics.append([query_id, precision, recall, ap, time])

# overall_metrics.get_avg_metrics_over_query_set()

# print("Storing Metrics")
# overall_metrics.store_metrics_to_csv()

# print("Done!")


#Binary Independence Model
# binary_independence_model = BinaryIndependenceModel(known_relevant_docs_cnt=20)
# binary_independence_model.load_index_from_csv()
# print("loading index done")

# # load all queries and relevant docs for them
# relevant_docs_per_query = load_relevant_docs_for_each_query(binary_independence_model.listdir)
# queries = load_all_queries()
# print("loading queries done")


# print("Running Queries")
# overall_metrics = EvaluationMetrics(query_set_length=len(queries), query_set_metrics=[], file_path="metrics_bim.csv")
# for query_id in queries.keys():
#     query = Query(query_id, queries[query_id], relevant_docs_per_query[query_id])
#     t = timeit.repeat(lambda: binary_independence_model.return_relevant_docs_for_query(query.query,query.relevant_docs), number=1, repeat=1)
#     time = round(sum(t)/len(t),3)
#     retrieved_docs = binary_independence_model.return_relevant_docs_for_query(query.query,query.relevant_docs)
#     query.cnt_actual_relevant_docs_returned_and_ranks(retrieved_docs)
#     precision, recall, ap = query.evaluate_metrics(query.actual_relevant_docs_cnt, query.actual_relevant_docs_pos)
#     overall_metrics.query_set_metrics.append([query_id, precision, recall, ap, time])
#     # print(precision,recall,ap,time)
#     # break

# overall_metrics.get_avg_metrics_over_query_set()

# print("Storing Metrics")
# overall_metrics.store_metrics_to_csv()

# print("Done!")

