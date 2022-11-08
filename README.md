# Search-Engine
1. A simple search engine made using three different models: binary independence model, tf-idf vector model, language model and elastic search.
2. Implemented text classification using KNN and Rocchio algorithm.
3. Evaluated all the models using metrics like precision, recall, MAP and running time per query.

Please install the requirements.txt and softwares necessary for using elastic search before running the code.

# Instructions to run the Code:
Run this command to load all the required libraries and run the code
```
pip install -r requirements.txt
python main.py
```

**Note:** The elastic search evaluation given at the end of run.py will not run currently as it requires to deploy a elastice search instance and none is deployed currently. To make it work first deploy one and then replace its endpoint in the code so it can be accessed. 

The code for testing and evaluating different models is given in run.py. This first constructs a index on given set of documents and then evaluates each model on a set of queries and stores the metrics in separate files.
A separate class for query and evaluation metrics is created for modularity of code.
The results of each model consisting of precision, recall, map, running time per query and their averages have been stored in metrics_tfidf.csv, metrics_lm.csv, metrics_bim.csv and metrics_es.csv for vector model, language model, binary independence model and elasticsearch respectively.
The inverted_index is stored in inverted_index.csv also the document vector lengths have been stored.
The set of docs and queries can be found at https://drive.google.com/file/d/1e3JUIP2tSAzsJPNaV9Mkl2NyGk8CMEmI/view?usp=share_link
To run code keep Docs folder(without extra nesting within another Docs) should be in Search Engine folder as all code files otherwise the paths should be changed in the code. The Docs folder contains alldocs folder, query.txt, output.txt(alldocs is compressed as .tar so first extract it)

The best MAP=9.58 for pseudo-relevance feedback is given by alpha = 1 i.e. only consider the original query and ignore the centroid. The MAP increases as the alpha increases. The variance of map with alpha is given map_vs_alpha.txt file. 

Performance wise:
In terms of precision,recall,map: Language Model > ElasticSearch > TF-iDF > Binary Independence Model
In terms of running time per query: Binary Independence Model > TF-iDF > Elastic Search > Language Model

Results are given below:

| Model                      | TF-IDF | ElasticSearch | LanguageModel | BinaryIndependenceModel |
|----------------------------|--------|---------------|---------------|-------------------------|
| avg_precision              | 0.921  | 0.979         | 0.995         | 0.626                   |
| avg_recall                 | 0.223  | 0.238         | 0.243         | 0.152                   |
| map                        | 0.958  | 0.991         | 0.998         | 0.674                   |
| avg_running_time_per_query | 0.030  | 0.054         | 0.078         | 0.008                   |
| max_running_time_per_query | 0.077  | 0.212         | 0.140         | 0.023                   |
| min_running_time_per_query | 0.005  | 0.015         | 0.030         | 0.002                   |

## Text Classification using KNN and Rocchio
The results with precision, recall and f1 score for knn(k=1,3,5) and Rocchio are given in document_classification in respective test files

