# Search-Engine
1. A simple search engine made using three different models: binary independence model, tf-idf vector model, language model and elastic search.
2. Implemented text classification using KNN and Rocchio algorithm.
3. Evaluated all the models using metrics like precision, recall, MAP and running time per query.

Please install the requirements.txt before running the code.
Q3 part 1a is done in inverted_index.py, utils.py, vector_model.py
Q3 part 1b is done in 1b.py and elastic.py
Q3 part 2 pseudo relevance feedback is implemented in vector_model.py
Q3 part 3 tdf-idf already implemented in part1a, binary independence model in binary_independence_model.py and language model in language_model.py
Q3 part 4 All code files and results in q3_4 folder
# Instructions to run the Code:
Run this command to load all the required libraries and run the code
pip install requirements.txt
python main.py

The code for testing and evaluating 1a,2,3 is given in run.py. Uncomment the corresonding part and model code to test it.
The code for testing and evaluating 1b is in 1b.py.
Also, it is recommended that you first create a index locally on your pc before testing. For this uncomment the 2 lines code below the comment "Create inverted index for the docs" on line 11. 
A separate class for query and evaluation metrics is created for modularity of code.
The results of each model consisting of precision, recall, map, running time per query and their averages have been stored in metrics_tfidf.csv metrics_lm.csv and metrics_bim.csv respectively and metrics_1b.csv for ElasticSearch.
The inverted_index created in part 1a is stored in inverted_index.csv also the document vector lengths have been stored.
To run code keep alldocs folder, query.txt, output.txt should be in same folder as all code files otherwise the paths should be changed in the code.

# Part-2
The best MAP=9.58 in part 2 is given by alpha = 1 i.e. only consider the original query and ignore the centroid. The MAP increases as the alpha increases. The variance of map with alpha is given map_vs_alpha.txt file. 

# Part-1a,1b,3
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

# Part-4 (Text Classification using KNN and Rocchio)
The results with precision, recall and f1 score for knn(k=1,3,5) and Rocchio are given in q3_4 in respective test files

