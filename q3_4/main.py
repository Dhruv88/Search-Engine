# import the required libraries
from inverted_index import InvertedIndex
from vector_model import VectorModel
from utils import read_text_file
from collections import defaultdict,Counter
import collections

import sklearn.metrics


def load_dataset(file_path = 'output.txt'):
    relevant_docs_query = {} # dictionary to store the relevant documents for each query
    with open(file_path, 'r', encoding="utf8") as f:
        while True:
            line = f.readline() # read the next line
            if line == "": # end of file is reached
                break
            if line == "\n": # skip empty lines
                continue
            query,doc = line.split()[0:2] # split the line into query and document
            query = int(query) # convert the query to integer
            if query not in relevant_docs_query:
                relevant_docs_query[query] = set() # create a set to store the relevant documents for the query
            if query in relevant_docs_query:
                relevant_docs_query[query].add(doc)
    return relevant_docs_query

def remove_repeated_relevant_docs(relevant_docs_query):
    # remove the docs from the doc_set in relevant_docs_query which are occuring more than once
    relevant_doc_set = set()
    remove_doc_set = set()
    for query,doc_set in relevant_docs_query.items():
        for doc in doc_set:
            # add the doc to the relevant_doc_set if it didn't already exist
            if doc not in relevant_doc_set:
                relevant_doc_set.add(doc)
            # remove the doc from the doc_set and relevant_docs_query if it already exists
            else:
                remove_doc_set.add(doc) # add the doc to the remove_doc_set
    
    # remove the docs in the remove_doc_set from the relevant_docs_query
    for query,doc_set in relevant_docs_query.items():
        for doc in remove_doc_set: # iterate over the remove_doc_set
            if doc in doc_set: # if the doc is in the doc_set
                doc_set.remove(doc) # remove the doc from the doc_set
    return relevant_docs_query,remove_doc_set # return the relevant_docs_query and the remove_doc_set

def train_test_split(relevant_docs_query, test_size = 0.3):
    # split the dataset into training and test sets with 30% of the data used for testing without using model_selection
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    # Randomly select 70% of the relevant docs for training and 30% for testing
    for query,doc_set in relevant_docs_query.items():
        size_counter=0 # counter to keep track of the number of docs in the doc_set
        for doc in doc_set: 
            if size_counter < (1-test_size)*len(doc_set): # if the counter is less than 70% of the size of the doc_set
                X_train.append(doc) # add the doc to the training set
                y_train.append(query)
            else:
                X_test.append(doc) # add the doc to the test set
                y_test.append(query)
            size_counter+=1 # increment the counter
    return X_train, X_test, y_train, y_test


def train_test_split_2(relevant_docs_query,X_test,X_train):
    Y_train,Y_test=[],[]
    for doc in X_train:
        # find the doc in relevant_docs_query and add the corrosponding query to Y_train
        for query,doc_set in relevant_docs_query.items():
            if doc in doc_set:
                Y_train.append(query)
    for doc in X_test:
        # find the doc in relevant_docs_query and add the corrosponding query to Y_test
        for query,doc_set in relevant_docs_query.items():
            if doc in doc_set:
                Y_test.append(query)
    return Y_train,Y_test


def docID_docName_set(vm, X_train, y_train): 
    docID_docName_dict = defaultdict(list) # dictionary to store the docID and docName
    for i in range(len(X_train)): # iterate over the training set
        if X_train[i] in vm.listdir: # if the docID is in the list of docIDs
            docID_docName_dict[y_train[i]].append([vm.listdir.index(X_train[i]),X_train[i]]) # add the docID and docName to the dictionary
    return docID_docName_dict

def rochhio(X_train,X_test,y_train,y_test):
    vm = VectorModel(InvertedIndex(index_file_path='inverted_index_clf.csv'),docs_to_be_retrieved=5) # create a vector model
    vm.load_index_from_csv() # load the index from csv file
    vm.load_doc_lengths_from_csv() # load the doc lengths from csv file
    # Rochhio
    docID_docName = docID_docName_set(vm, X_train, y_train) # get the docID_docName dictionary
    print(sorted(docID_docName.keys()))
    centroids_class = defaultdict(Counter) # dictionary to store the centroids for each class
    cnt = 0
    for cls, doc_list in docID_docName.items():
        # print(cls,doc_list)
        centroids_class[cls]=vm.calc_centroid(doc_list) # calculate the centroid for each class
        cnt+=1
        if cnt%10==0:
            print(cnt,"centroids done")
    # print(centroids_class)
    print("done")
    y_pred = [0]*len(X_test) # list to store the predicted labels
    i = 0
    for doc in X_test:
        centroid_dist = defaultdict(lambda: 0.0) # dictionary to store the distance of the doc from each centroid
        if doc not in vm.listdir:
            continue
        doc_path = './alldocs/'+doc # get the path of the doc
        doc_content = read_text_file(doc_path) # read the content of the doc
        # doc_vector = vm.get_token_cnt(doc_content) # get the query vector for the doc using Manhattan Distance
        doc_vector = vm.make_query_vector(doc_content) # get the query vector for the doc using Euclidean Distance
        min_dist = float('inf') # initialize the minimum distance to infinity
        for cls in centroids_class.keys(): # for each class
            ttl_terms = set(centroids_class[cls].keys()).union(set(doc_vector.keys())) # get the set of terms in the centroid using Euclidean Distance
            # ttl_terms = doc_vector + centroids_class[cls] # Manhattan distance
            dist = 0.0
            for term in ttl_terms: # for each term in the centroid or the doc
                x1,x2 = 0.0,0.0 # initialize the x1 and x2 to 0
                if term in doc_vector: # if the term is in the doc
                    x1 = doc_vector[term] # set x1 to the value of the term in the doc

                if term in centroids_class[cls]:  # if the term is in the centroid
                    x2 = centroids_class[cls][term]
                # dist += abs(x1 - x2) #Manhattan DIstance
                centroid_dist[cls] += ((x1-x2)**2) # calculate the euclidean distance of the doc from the centroid
            if min_dist > dist: # if the distance is less than the minimum distance
                min_dist = dist # set the minimum distance to the distance
                y_pred[i] = cls # set the predicted label to the class
        i+=1
        if i%100==0:
            print(i,"docs done")
    print('\n\nClassification report for Rochhio classifier:') # print the classification report
    print(sklearn.metrics.classification_report(y_test, y_pred))
    print('\n\n')

def KNN(X_train, X_test, y_train, y_test,k=5):
    vm = VectorModel(InvertedIndex(index_file_path='inverted_index_clf.csv'),docs_to_be_retrieved=k) # create a vector model
    vm.load_index_from_csv() # load the index from csv file
    vm.load_doc_lengths_from_csv() # load the doc lengths from csv file
    k_nearest_docs={}
    for doc in X_test:
        if doc not in vm.listdir: # if the doc is not in the list of docs
            continue
        # print(doc)
        doc_path = './alldocs/'+doc # get the path of the document
        doc_content = read_text_file(doc_path) # read the document
        # print(doc_content)
        if type(doc_content) != str:
            print(doc_content,'this doc is not str')

        k_nearest_docs[doc] = vm.return_top_k_nearest_docs(doc=doc_content,ignore=ignore_docs) # get the top k nearest docs
       
    class_predicted_dict={}
    class_predicted=[] # list to store the predicted labels
    for doc,predicted_doc_list in k_nearest_docs.items(): # for each doc in the test set
        temp_lst=[]
        for predicted_doc in predicted_doc_list: # for each of the k nearest docs
            if predicted_doc[1] in X_train:
                temp_lst.append(y_train[X_train.index(predicted_doc[1])]) # get the class of the doc
        class_predicted_dict[doc]=temp_lst # get the predicted class for each doc
        class_predicted.append(collections.Counter(temp_lst).most_common(1)[0][0]) # get the most common class

    print()
    # print(class_predicted)
    l=len(class_predicted)
    print(f'\n\nClassification report for KNN classifier for k={k}:') # print the classification report
    print(sklearn.metrics.classification_report(y_test[0:l], class_predicted)) # print the classification report
    print('\n\n')



# main begins here
relevant_docs_query = load_dataset('output.txt')   # load the dataset
# print(relevant_docs_query)
relevant_docs,removed = remove_repeated_relevant_docs(relevant_docs_query) # remove the repeated relevant docs
# print(relevant_docs)
print()
# X_train, X_test, y_train, y_test = train_test_split(relevant_docs,0.3) # split the relevant docs into training and test sets

ignore_docs=set()
ignore_docs = removed

# print(ignore_docs)
inv_i=InvertedIndex() # create an empty inverted index
# inv_i.process_all_documents(ignore=ignore_docs) # ignore the docs that are already in the training set


X_train,X_test = inv_i.prepare_datasets_2()
X_test = set(X_test)
X_test = list(X_test.difference(ignore_docs))
y_train,y_test = train_test_split_2(relevant_docs,X_test,X_train)
rochhio(X_train,X_test,y_train,y_test)
# print(X_train[15:20], y_train[15:20])
KNN(X_train, X_test, y_train, y_test,k=1) # k=1
KNN(X_train, X_test, y_train, y_test,k=3) # k=3
KNN(X_train, X_test, y_train, y_test,k=5) # k=5
