import requests
import matplotlib.pyplot as plt
import seaborn as sns
import math
import collections
from myPreProcessor import MyPreProcessor
from myDBHandler import MyDBHandler
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.decomposition import PCA, TruncatedSVD
from scipy.sparse import csr_matrix
from imblearn.over_sampling import SVMSMOTE
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from sklearn.datasets import make_classification
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import datasets

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


def distinct_features_labels(_allX, _allCategories):
    print("___________________________________________________________________________________________________")
    _XTemp = []
    _yTemp = []
    _i = 0
    print("All Feature Indexes::" + str(len(_allX)))
    for _index, _item in enumerate(_allX):
        # if _item not in _XTemp and _allCategories[_index] != "Medical":
        if _item not in _XTemp:
            _XTemp.append(_item)
            _yTemp.append(_allCategories[_index])
        else:
            _i += 1
    print("Feature Indexes Excluded::" + str(_i))
    return [_XTemp, _yTemp]


def data_manipulation(_list, _joinString, _isExtended):
    _listManipulated = []
    if _isExtended:
        for _item in _list:
            _item = preProcessor.data_pre_processing_nltk_extended(_item)
            _toSpaceString = _joinString.join(_item)
            _listManipulated.append(_toSpaceString)
    else:
        for _item in _list:
            _item = preProcessor.data_pre_processing_nltk_extended(_item)
            _toSpaceString = _joinString.join(_item)
            _listManipulated.append(_toSpaceString)
    return _listManipulated


def data_printer(_list, _listName):
    print("___________________________________________________________________________________________________")
    print(_listName + " - Data printing: ")
    _counter = collections.Counter(_list)
    # print(_list)
    print(_counter)


def select_n_components(_var_ratio, _goal_var):
    _total_variance = 0.0
    _n_components = 0
    for explained_variance in _var_ratio:
        _total_variance += explained_variance
        _n_components += 1
        if _total_variance >= _goal_var:
            break
    return _n_components


def data_confusion_matrix(_y_test, _y_predicted):
    print("___________________________________________________________________________________________________")
    _y_test_set = sorted(set(_y_test))
    _y_test_size = len(_y_test_set)
    _y_predicted_set = sorted(set(_y_predicted))
    _y_predicted_size = len(_y_predicted_set)
    print("Tested Label's size for confusion_matrix: ", _y_test_size)
    print("Predicted Label's size for confusion_matrix: ", _y_predicted_size)

    _CM = confusion_matrix(_y_test, _y_predicted)

    if _y_test_size > _y_predicted_size:
        sns.heatmap(_CM, cmap="Reds", linewidths=0.1, annot=True, cbar=False,
                    xticklabels=_y_test_set, yticklabels=_y_test_set, annot_kws={"size": 8},
                    fmt="d")
    else:
        sns.heatmap(_CM, cmap="Reds", linewidths=0.1, annot=True, cbar=False,
                    xticklabels=_y_predicted_set, yticklabels=_y_predicted_set, annot_kws={"size": 8},
                    fmt="d")

    _bottom, _top = plt.ylim()
    _bottom += 0.5
    _top -= 0.5
    plt.ylim(_bottom, _top)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted output")
    plt.ylabel("True output")
    plt.show()

    _FP = 0
    _FN = 0
    _TP = 0
    _TN = 0
    _PRECISION = 0
    _cm_length = int(math.sqrt(_CM.size))
    for _i in range(_cm_length):
        _tFP = 0
        _tFN = 0
        _tTP = 0
        _tTN = 0
        for _j in range(_cm_length):
            if _j == _i:
                _TP += _CM[_i][_j]
                _tTP += _CM[_i][_j]
            else:
                _FN += _CM[_i][_j]
                _tFN += _CM[_i][_j]
                _FP += _CM[_j][_i]
                _tFP += _CM[_j][_i]
        if 0 == (_tTP + _tFP):
            _PRECISION = 0
        else:
            _PRECISION += _tTP / (_tTP + _tFP)

    print("My micro Precision: ", round(_TP * 100 / (_TP + _FP), 5))
    print("My macro Precision: ", round(_PRECISION * 100 / _cm_length, 5))


def result_handler(_pipeline, _pipelineName, _x_train, _x_test, _y_train, _y_test):
    print("___________________________________________________________________________________________________")
    print(_pipelineName + " - Result printing: ")
    _pipeline.fit(_x_train, _y_train)
    _y_predicted = _pipeline.predict(_x_test)

    data_printer(_y_predicted, "y_predicted")
    data_confusion_matrix(_y_test, _y_predicted)

    _accuracy = round(accuracy_score(_y_test, _y_predicted) * 100, 5)
    _recall = round(recall_score(_y_test, _y_predicted, average="macro") * 100, 5)
    _precision_macro = round(precision_score(_y_test, _y_predicted, average="macro") * 100, 5)
    _precision_micro = round(precision_score(_y_test, _y_predicted, average="micro") * 100, 5)
    _f1 = round(f1_score(_y_test, _y_predicted, average="macro") * 100, 5)
    print("Accuracy: ", _accuracy)
    print("Recall: ", _recall)
    print("Precision micro: ", _precision_micro)
    print("Precision macro: ", _precision_macro)
    print("F1: ", _f1)
    return _f1


def le_cosine(_X, _y, _yFactor):
    _cosine_distance = 1 - cosine_similarity(_X)
    _json = {}
    for _i_index, _item in enumerate(_cosine_distance):
        _json[str(_i_index) + "FP"] = ""
        _json[str(_i_index) + "FN"] = ""
        for _j_index, _j in enumerate(_item):
            if _j < 0.66:
                _json[str(_i_index) + "FP"] += str(_y[_j_index]) + ","
            if _j > 0.99:
                _json[str(_i_index) + "FN"] += str(_y[_j_index]) + ","

    _exclude = []
    _counter_able = ""
    _fp = 0
    _pair_index = 0
    limit = int(_yFactor/2)
    for _index, _item in enumerate(_json):
        if _index % 2 == 0:
            _pair_index = _index
            _fp = _json[_item].split(",")
            _counter_able = _fp[0]
            _fp = len(set(_fp))
        else:
            _fn = _json[_item].split(",")
            _counter = collections.Counter(_fn)
            if _fp > limit or counter[_counter_able] > limit:
                _exclude.append(int(_pair_index/2))

    _XCosine = np.delete(_X.toarray(), _exclude, 0)
    _yCosine = np.delete(_y, _exclude, 0)
    print("Cosine similarity Indexes Excluded::" + str(len(_exclude)))
    return _XCosine, _yCosine


if __name__ == "__main__":

    try:
        FETCH4STORE = True
        ML = False
        FETCH4STORE = False
        ML = True
        # --------------------------
        # QUERY = "bioengineering"
        # QUERY = "biophysics"
        # QUERY = "bioeconomics"
        # QUERY = "biology"
        # QUERY = "zoology"
        # QUERY = "microbiology"
        # QUERY = "immunology"
        # QUERY = "genetics"
        # QUERY = "forensic"
        # QUERY = "ecology"
        # QUERY = "botany"
        # QUERY = "bioethics"
        # QUERY = "biodiversity"
        QUERY = ""

        DB = "gbooks"
        # QUERY = "biochemistry"
        # QUERY = "bioelectrical"
        # QUERY = "bioimpedance"
        # QUERY = "bioinformatics"
        # QUERY = "biomaterials"
        # QUERY = "biomedical"
        # QUERY = "biomedicine"
        # QUERY = "biostatistics"
        # QUERY = "biotechnology"
        # --------------------------
        # QUERY = "body composition"
        # QUERY = "lymphedema"
        # --------------------------
        # QUERY = "bioresistance"
        # QUERY = "bioreactance"
        # --------------------------
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
            # PARAMS = {"q": QUERY, "key": "AIzaSyD3etuN-ZjEVM85annR8Gc0sb1Z3phWCqA"}
            # PARAMS = {"q": QUERY, "key": "AIzaSyA3T_IBqnf4zkWjlEjMj4NomvoCf-P-nFE"}
            # PARAMS = {"q": QUERY, "key": "AIzaSyCdZPzYO3cZg89P7ATWPxcQSNYpkRmNCls"}
            # PARAMS = {"q": QUERY, "key": "AIzaSyBo90b_VuOYcgbiD9hFIWXdDLwweSwtQLA"}
            # PARAMS = {"q": QUERY, "key": "AIzaSyCsv_2gdLYhpKnz3CdGDykB4HZB-ZZKIsE"}
            # PARAMS = {"q": QUERY, "key": "AIzaSyCc-2TofUelwwh5sRNW4Uz-EgWStfM_jqw"}
            # PARAMS = {"q": QUERY, "key": "AIzaSyBT7Y4p9r8PIWqmoP4sJnULGGUJCafUesY"}
            # PARAMS = {"q": QUERY, "key": "AIzaSyD_daqJ7ToX5BIcaLa5EL71PsM8hAqrxr8"}
            # PARAMS = {"q": QUERY, "key": "AIzaSyDcgDVEHAAETDM4cArfA_QIGfqu4BWPHtg"}
            PARAMS = {"q": QUERY, "key": "AIzaSyBUTnJnda81h7AJ3pmFUB519RtDigzHulo"}

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
                    if "error" in request_data and request_data["error"]["code"] == 403:
                        raise Exception(request_data)
                    else:
                        next_page = False
                start_index += 1
            print("Fetching data for db-storing performed successfully!")

        if ML:
            print("Machine Learning started ...")
            all = databaseHandler.get_all_data([
                "data_biochemistry",
                "data_biodiversity",
                "data_bioeconomics",
                "data_bioelectrical",
                "data_bioengineering",
                "data_bioethics",
                "data_bioimpedance",
                "data_bioinformatics",
                "data_biology",
                "data_biomaterials",
                "data_biomedical",
                "data_biomedicine",
                "data_biophysics",
                "data_biostatistics",
                "data_biotechnology",
                "data_body composition",
                "data_botany",
                "data_ecology",
                "data_forensic",
                "data_genetics",
                "data_immunology",
                "data_lymphedema",
                "data_microbiology",
                "data_zoology",
                "data_abnormal",
                "data_abortion",
                "data_absorption",
                "data_acute",
                "data_adaptation",
                "data_agent",
                "data_allele",
                "data_allergen",
                "data_allergy",
                "data_amniocentesis",
                "data_anthrax",
                "data_antibiotic",
                "data_antibodies",
                "data_antibody",
                "data_antidote",
                "data_antigen",
                "data_antioxidant",
                "data_antiviral",
                "data_arteriosclerosis",
                "data_ash",
                "data_assembler",
                "data_atom",
                "data_biopower",
                "data_biorefinery",
                "data_bioregion",
                "data_bioremediation",
                "data_biota",
                "data_biotechnology",
                "data_bioterrorism",
                "data_blastocyst",
                "data_botanical",
                "data_botulism",
                "data_brucellosis",
                "data_buckminsterfullerene"
            ], "item")
            # all = databaseHandler.get_all_data([
            #     "data_biochemistry",
            #     "data_biodiversity",
            #     "data_bioethics",
            #     "data_biology",
            #     "data_biomedical",
            #     "data_biophysics",
            #     "data_biotechnology",
            #     "data_botany",
            #     "data_ecology",
            #     "data_forensic",
            #     "data_genetics",
            #     "data_immunology",
            #     "data_microbiology",
            #     "data_zoology"
            # ], "item")
            allX = all[0]
            allCategories = all[1]

            distinctXyTemp = distinct_features_labels(allX, allCategories)
            XTemp = distinctXyTemp[0]
            yTemp = distinctXyTemp[1]

            # # Load text data.
            # textData = datasets.fetch_20newsgroups(remove=("headers", "footers", "quotes"))
            #
            # # Store features and target variable into "X" and "y".
            # XTemp = textData.data
            # yTemp = textData.target

            XTempManipulated = data_manipulation(XTemp, " ", True)
            yTemp = data_manipulation(yTemp, "", False)

            data_printer(yTemp, "yTemp")

            labelEncoder = LabelEncoder()
            yTempManipulated = labelEncoder.fit_transform(yTemp)

            data_printer(yTempManipulated, "yTempManipulated")
            counter = collections.Counter(yTempManipulated)
            otherLabelEncoded = max(yTempManipulated) + 1
            print("Other category label::", otherLabelEncoded)

            X = []
            y = []
            i = 0
            for index, item in enumerate(yTempManipulated):
                yCount = counter[item]
                yFactor = yCount/len(yTempManipulated)
                if yFactor > 0.04:
                    y.append(yTempManipulated[index])
                    X.append(XTempManipulated[index])
                    i += 1
                # else:
                #     y.append(otherLabelEncoded)
                #     X.append(XTempManipulated[index])
            print("Xy Indexes Included::" + str(i))

            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(X)

            # minMaxScaler = QuantileTransformer()
            # X = minMaxScaler.fit_transform(X)

            X, y = le_cosine(X, y, 4)

            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
            # y = np.array(y)
            folds = 1
            # kf = KFold(n_splits=folds)
            sumF1MultinomialNB = 0
            sumF1LogisticRegression = 0
            sumF1SVC = 0
            # for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
            #     x_train = X[train_index]
            #     y_train = y[train_index]
            #     x_test = X[test_index]
            #     y_test = y[test_index]

            # print("x_train shape::" + str(x_train.shape))
            # data_printer(y_train, "y_train")
            #
            # nm1 = RandomUnderSampler(replacement=True)
            # x_train, y_train = nm1.fit_resample(x_train, y_train)
            #
            # print("x_train shape::" + str(x_train.shape))
            # data_printer(y_train, "y_train")

            sm = SVMSMOTE()
            x_train, y_train = sm.fit_resample(x_train, y_train)

            # tsvd = TruncatedSVD(n_components=maxSpace - 1)
            # tsvd.fit(X)
            # n_components = select_n_components(tsvd.explained_variance_ratio_, 0.95)
            #
            # print("truncatedSVD space::" + str(n_components))
            # truncatedSVD = TruncatedSVD(n_components=n_components)
            # # truncatedSVD = PCA(n_components=minSpace)
            # X = truncatedSVD.fit_transform(X)

            data_printer(y_train, "y_train")
            data_printer(y_test, "y_test")

            # pipelineMultinomialNB = MultinomialNB(alpha=0.1)
            pipelineLogisticRegression = LogisticRegression(solver="lbfgs", multi_class="multinomial",
                                                            class_weight="balanced")
            pipelineSVC = SVC(C=1, kernel="linear", class_weight="balanced")

            # pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB(alpha=0.1))
            # pipeline = make_pipeline(TfidfVectorizer(), SVC(C=5, kernel="linear", class_weight="balanced"))
            # pipeline = make_pipeline(TfidfVectorizer(), AdaBoostClassifier(LinearSVC(), n_estimators=1000, learning_rate=1.0, algorithm="SAMME"))
            # pipeline = AdaBoostClassifier(n_estimators=100, base_estimator=pipeline, learning_rate=1)
            # pipeline = BaggingClassifier(base_estimator=pipeline)
            # estimators = [
            #     ("ml", BaggingClassifier(base_estimator=MultinomialNB(alpha=0.1))),
            #     ("bsvm", BaggingClassifier(base_estimator=SVC(C=1, kernel="linear", class_weight="balanced")))
            # ]
            # pipeline = VotingClassifier(estimators)

            # sumF1MultinomialNB += result_handler(pipelineMultinomialNB, "MultinomialNB", x_train, x_test, y_train, y_test)
            sumF1LogisticRegression += result_handler(pipelineLogisticRegression, "LogisticRegression", x_train, x_test, y_train, y_test)
            sumF1SVC += result_handler(pipelineSVC, "SVC", x_train, x_test, y_train, y_test)

            # print("SVC F1: ", round(sumF1SVC/folds, 5))
            # print("LogisticRegression F1: ", round(sumF1LogisticRegression/folds, 5))

        print("Pause")
    except BaseException as error:
        print("===================================================================================================")
        print("Error on_main: %s" % str(error))
        print("===================================================================================================")

