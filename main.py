import requests
import matplotlib.pyplot as plt
import seaborn as sns
import math
import collections
from myPreProcessor import MyPreProcessor
from myDBHandler import MyDBHandler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


# Extra Comments
#
# git config --replace-all user.name "margolleoAUTH"
# git config --replace-all user.email "margolleo@csd.auth.com"
# git commit --author="margolleoAUTH <margolleo@csd.auth.com>"
#
# Database(Atlas):
# https://cloud.mongodb.com/user?n=%2Fv2%2F5ce5e9a1ff7a252fdb34b8b1&nextHash=%23clusters#/atlas/login
# margolleo@csd.auth.com
# 5Kleanthis@
# Please add your IP to the whitelist in order to connect with the database(We do not have open firewall - 0.0.0.0)
# preprocessor lib does not work for pip install - needs to be installed manually

if __name__ == "__main__":

    try:
        FETCH4STORE = True
        ML = False
        FETCH4STORE = False
        ML = True
        DB = "gbooks"
        # QUERY = "bioimpedance"
        # QUERY = "bioelectrical"
        # QUERY = "body composition"
        # QUERY = "lymphedema"
        QUERY = "body fluid"
        DATA_COL = "data_" + QUERY
        CAT_COL = "categories_" + QUERY

        # Initialization of Controller's Instances
        # Database Handler
        # Pre-Processor
        preProcessor = MyPreProcessor()
        databaseHandler = MyDBHandler(DB)

        if FETCH4STORE:
            print("Fetching data for db-storing started ...")
            databaseHandler.delete_all_data(DATA_COL)
            databaseHandler.delete_all_data(CAT_COL)

            # API-Endpoint
            URL = "https://www.googleapis.com/books/v1/volumes"
            PARAMS = {"q": QUERY, "key": "AIzaSyA3T_IBqnf4zkWjlEjMj4NomvoCf-P-nFE"}

            X = []
            next_page = True
            start_index = 0
            while next_page:
                PARAMS["startIndex"] = start_index
                request_data = requests.get(url=URL, params=PARAMS).json()
                if "items" in request_data:
                    data = request_data["items"]
                    for item in data:
                        if "title" in item["volumeInfo"] and "description" in item["volumeInfo"] and "categories" in \
                                item["volumeInfo"]:
                            item_data = item["volumeInfo"]["title"]
                            item_data += " " + item["volumeInfo"]["description"]
                            item_category = item["volumeInfo"]["categories"][0]
                            if item_data not in X:
                                X.append(item_data)
                                databaseHandler.insert_data(DATA_COL, {"item": item_data})
                                databaseHandler.insert_data(CAT_COL, {"item": item_category})
                else:
                    next_page = False
                start_index += 1
            print("Fetching data for db-storing performed successfully!")

        if ML:
            print("Machine Learning started ...")
            allX = databaseHandler.get_all_data("data_bio", "item")
            allCategories = databaseHandler.get_all_data("categories_bio", "item")

            XTemp = []
            yTemp = []
            i = 0
            for index, item in enumerate(allX):
                if item not in XTemp:
                    XTemp.append(item)
                    yTemp.append(allCategories[index])
                else:
                    i += 1
            print("Indexes Excluded::" + str(i))

            XTempManipulated = []
            for item in XTemp:
                item = preProcessor.data_pre_processing_nltk_extended(item)
                toSpaceString = " ".join(item)
                XTempManipulated.append(toSpaceString)

            labelEncoder = LabelEncoder()
            yTempManipulated = labelEncoder.fit_transform(yTemp)

            counter = collections.Counter(yTempManipulated)
            print("Frequencies of distinct categories: ")
            print(counter)

            X = []
            y = []
            otherLabelEncoded = max(yTempManipulated) + 1
            for index, item in enumerate(yTempManipulated):
                yCount = counter[item]
                yFactor = yCount/len(yTempManipulated)
                if yFactor > 0.1:
                    y.append(yTempManipulated[index])
                else:
                    y.append(otherLabelEncoded)
                X.append(XTempManipulated[index])

            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

            print("x_train:")
            print(collections.Counter(x_train))
            print("___________________________________________________________________________________________________")
            print("x_test:")
            print(collections.Counter(x_test))

            print("y_train:")
            print(y_train)
            print(collections.Counter(y_train))
            print("___________________________________________________________________________________________________")
            print("y_test:")
            print(y_test)
            print(collections.Counter(y_test))

            # pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB(alpha=0.1))
            pipeline = make_pipeline(TfidfVectorizer(), SVC(C=1, kernel="linear", gamma="auto"))
            pipeline.fit(x_train, y_train)

            y_predicted = pipeline.predict(x_test)

            print("___________________________________________________________________________________________________")
            print("y_predicted:")
            print(y_predicted)
            print(collections.Counter(y_predicted))

            y_predicted_set = sorted(set(y_predicted))
            y_predicted_size = len(y_predicted_set)
            y_test_set = sorted(set(y_test))
            y_test_size = len(y_test_set)
            print("Predicted Label's size: ", y_predicted_size)
            print("Tested Label's size: ", y_test_size)

            CM = confusion_matrix(y_test, y_predicted)

            if y_test_size > y_predicted_size:
                sns.heatmap(CM, cmap="Reds", linewidths=0.1, annot=True, cbar=False,
                            xticklabels=y_test_set, yticklabels=y_test_set, annot_kws={"size": 8},
                            fmt="d")
            else:
                sns.heatmap(CM, cmap="Reds", linewidths=0.1, annot=True, cbar=False,
                            xticklabels=y_predicted_set, yticklabels=y_predicted_set, annot_kws={"size": 8},
                            fmt="d")

            bottom, top = plt.ylim()
            bottom += 0.5
            top -= 0.5
            plt.ylim(bottom, top)
            plot_title = "Confusion Matrix"
            plt.title(plot_title)
            plt.xlabel("Predicted output")
            plt.ylabel("True output")
            plt.show()

            FP = 0
            FN = 0
            TP = 0
            TN = 0
            PREC = 0
            cm_length = int(math.sqrt(CM.size))
            for i in range(cm_length):
                tFP = 0
                tFN = 0
                tTP = 0
                tTN = 0
                for j in range(cm_length):
                    if j == i:
                        TP += CM[i][j]
                        tTP += CM[i][j]
                    else:
                        FN += CM[i][j]
                        tFN += CM[i][j]
                        FP += CM[j][i]
                        tFP += CM[j][i]
                if 0 == (tTP + tFP):
                    PREC = 0
                else:
                    PREC += tTP/(tTP + tFP)

            print("my micro Precision: ", round(TP*100/(TP + FP), 5))
            print("my macro Precision: ", round(PREC*100/cm_length, 5))

            accuracy = round(accuracy_score(y_test, y_predicted)*100, 5)
            recall = round(recall_score(y_test, y_predicted, average="macro")*100, 5)
            precision_macro = round(precision_score(y_test, y_predicted, average="macro")*100, 5)
            precision_micro = round(precision_score(y_test, y_predicted, average="micro")*100, 5)
            f1 = round(f1_score(y_test, y_predicted, average="macro")*100, 5)
            print("Accuracy: ", accuracy)
            print("Recall: ", recall)
            print("Precision micro: ", precision_micro)
            print("Precision macro: ", precision_macro)
            print("F1: ", f1)

        print("Pause")
    except BaseException as error:
        print("===================================================================================================")
        print("Error on_main: %s" % str(error))
        print("===================================================================================================")

