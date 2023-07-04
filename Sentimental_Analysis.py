import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import requests
from bs4 import BeautifulSoup

def sentimental_analysis(input_dict, deposit_amount):
    dataset = pd.read_csv('Smart Bridge - Stock Analysis - Dataset.csv')
    
    # Preprocessing: Clean the text data and convert sentiment labels to numerical values
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    def preprocess_text(text):
        # Tokenize the text
        tokens = word_tokenize(text)

        # Remove stop words and perform stemming
        preprocessed_tokens = [stemmer.stem(token.lower()) for token in tokens if token.lower() not in stop_words]

        # Join the tokens back into a single string
        preprocessed_text = ' '.join(preprocessed_tokens)

        return preprocessed_text
    
    dataset['preprocessed_news'] = dataset['News'].apply(preprocess_text)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(dataset['preprocessed_news'], dataset['Sentiment'], test_size=0.2, random_state=42)

    # Feature extraction: Convert text to numerical features using TF-IDF
    tfidf = TfidfVectorizer()
    X_train_features = tfidf.fit_transform(X_train)
    X_test_features = tfidf.transform(X_test)
    
    print("For Sentimental Analysis of Data")
    
    # Logistic Regression
    logreg_model = LogisticRegression()
    logreg_model.fit(X_train_features, y_train)
    logreg_pred = logreg_model.predict(X_test_features)
    logreg_accuracy = accuracy_score(y_test, logreg_pred)
    print("Logistic Regression Accuracy:", logreg_accuracy)

    # Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train_features, y_train)
    nb_pred = nb_model.predict(X_test_features)
    nb_accuracy = accuracy_score(y_test, nb_pred)
    print("Naive Bayes Accuracy:", nb_accuracy)

    # Support Vector Machine
    svm_model = SVC()
    svm_model.fit(X_train_features, y_train)
    svm_pred = svm_model.predict(X_test_features)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    print("Support Vector Machine Accuracy:", svm_accuracy)

    # Random Forest
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train_features, y_train)
    rf_pred = rf_model.predict(X_test_features)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    print("Random Forest Accuracy:", rf_accuracy)

    # K-Nearest Neighbors
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train_features, y_train)
    knn_pred = knn_model.predict(X_test_features)
    knn_accuracy = accuracy_score(y_test, knn_pred)
    print("K-Nearest Neighbors Accuracy:", knn_accuracy)

    # Decision Tree
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train_features, y_train)
    dt_pred = dt_model.predict(X_test_features)
    dt_accuracy = accuracy_score(y_test, dt_pred)
    print("Decision Tree Accuracy:", dt_accuracy)
    
    # Define the URL of the website you want to scrape
    url = 'https://www.businesstoday.in/markets'

    # Send a GET request to the URL
    response = requests.get(url)

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the relevant HTML elements that contain the news articles
    articles = soup.find_all('ul', class_='mrk_buz_ul')

    # Iterate over the articles and extract the text
    news_list = []
    for article in articles:
        news_text = article.text.strip()  # Extract the text and remove leading/trailing whitespace
        news_list.append(news_text)
        
    m = news_list[0]
    new_news = []
    first = 0
    for i in range(len(m)-3):
        if(m[i]==" " and m[i+1]==" " and m[i+2]==" "):
            if(m[first:i] != " "):
                new_news.append(m[first:i].strip())
            first = i

    new_news.append(m[first:i])
    
    # Define the URL of the website you want to scrape
    url = 'https://economictimes.indiatimes.com/markets'

    # Send a GET request to the URL
    response = requests.get(url)

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the relevant HTML elements that contain the news articles
    articles = soup.find_all('div', class_='stry')

    # Iterate over the articles and extract the text
    news_list = []
    for article in articles:
        news_text = article.text.strip()  # Extract the text and remove leading/trailing whitespace
        news_list.append(news_text)
        
    new_news.extend(news_list)
    
    #print(new_news)
    
    # Predict sentiment for new news articles
    new_articles_preprocessed = [preprocess_text(article) for article in new_news]
    new_articles_features = tfidf.transform(new_articles_preprocessed)
    new_articles_sentiment = logreg_model.predict(new_articles_features)
    print("Predicted sentiment for new articles:", new_articles_sentiment)
    
    def compute_lps(pattern):
        """
        Compute the Longest Proper Prefix which is also a Suffix (LPS) array for the given pattern.
        """
        m = len(pattern)
        lps = [0] * m
        length = 0
        i = 1

        while i < m:
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1

        return lps


    def string_matching(text, pattern):
        """
        Perform string matching using the Knuth-Morris-Pratt (KMP) algorithm.
        Returns a list of matched patterns in the text.
        """
        n = len(text)
        m = len(pattern)
        lps = compute_lps(pattern)
        matches = []

        i = 0
        j = 0

        while i < n:
            if pattern[j] == text[i]:
                i += 1
                j += 1

                if j == m:
                    matches.append(pattern)
                    j = lps[j - 1]
            else:
                if j != 0:
                    j = lps[j - 1]
                else:
                    i += 1

        return matches


    # Example usage
    given_string = "This is a sample string containing TCS, Wipro, and Infosys."
    words = ["TCS", "Wipro", "Infosys", "Tech Mahindra", "HCL", "L&T", "Reliance", "HDFC", "ICICI", "SBI", "Axis",
             "Hindustan Unilever", "ITC", "Airtel", "Jio", "Kotak Mahindra", "TATA Steel", "TATA Motors",
             "Indian Oil Corporation", "Bharat Petroleum", "Bajaj", "ONGC", "Coal India", "NTPC", "Adani",
             "Ultratech", "Nestle", "Asian Paints", "Titan", "MRF", "Adani Ports", "Apollo Hospital", 
             "Bajaj Auto", "Bajaj Finance", "Britannia", "Cipla", "Divis Lab", "Dr Reddy's", "Eicher", 
            "Grasim", "HDFC LIFE", "HERO MotoCorp", "Hindalco", "IndusInd", "JSW Steel", "Maruti", 
            "Mahindra", "M&M", "M_M", "POWERGRID", "SBI LIFE", "Sun Pharma", "TATA CONSUM", "UPL", "DMART"]

    matched_words = []
    for given_string in new_news:
        x = []
        for word in words:
            if word in given_string:
                x.append(word)
        matched_words.append(x)

    #print("Matched Words:", matched_words)
    
    company = dataset["Company"]
    list_com = dataset["Company"].tolist()
    set_com = list(set(list_com))
    keys = set_com
    values = [0 for i in range(len(keys))]
    dictionary = dict(zip(keys, values))
    
    #print(dictionary)
    
    new_articles_sentiment = new_articles_sentiment.tolist()
    
    for i in range(len(matched_words)):
        for j in range(len(matched_words[i])):
            if(matched_words[i][j] in dictionary and new_articles_sentiment[i]=="Positive"):
                dictionary[matched_words[i][j]] += 0.4
            elif(matched_words[i][j] in dictionary and new_articles_sentiment[i]=="Negative"):
                dictionary[matched_words[i][j]] -= 0.4
            
    dict2 = dictionary
    
    dict1 = input_dict
    
    #print(dict2)
    #print(dict1)
    
    merged_dict = {}

    # Process keys present in dict1
    for key, value in dict1.items():
        if key in dict2:
            merged_dict[key] = (0.6 * value[0] , value[1]+ 0.4 * dict2[key])
        else:
            merged_dict[key] = (0.6 * value[0], value[1])

    # Process keys present in dict2 but not in dict1
    for key, value in dict2.items():
        if key not in dict1:
            merged_dict[key] = (0, 0.4 * value)

    #print(merged_dict)
    
    
    stocks = merged_dict

    converted_stocks = {}

    for key, value in stocks.items():
        converted_stocks[key] = {'price': value[0], 'rate': value[1]}

    #print(converted_stocks)
    
    available_amount = deposit_amount
    investments = {}

    print("Available Amount: " + str(available_amount))

    # Filter stocks with positive rate of change
    positive_stocks = {k: v for k, v in converted_stocks.items() if v['rate'] > 0}

    # Sort positive stocks based on potential profits
    sorted_stocks = sorted(positive_stocks.items(), key=lambda x: x[1]['price'] * x[1]['rate'], reverse=True)

    for stock, data in sorted_stocks:
        price = data['price']
        rate = data['rate']

        if price <= available_amount:
            max_stocks = available_amount // price  # Maximum number of stocks that can be bought
            invest_amount = price * max_stocks
            available_amount -= invest_amount
            investments[stock] = max_stocks


    # Print the investments
    for stock, quantity in investments.items():
        price = converted_stocks[stock]['price']
        print(f"Invest {quantity} stocks in {stock} at a price of {price}")

    # Print the leftover amount
    print(f"Leftover amount: {available_amount}")
    return dict(zip(dict1.keys(),new_articles_sentiment))

    
    
    
    
    
    
   
# dict1 = {'ADANIENT': (2441.4749, 1.1046422063939034),
#  'ADANIPORTS': (737.40704, -0.0532610463540123),
#  'APOLLOHOSP': (5162.115, 0.0012591896630137),
#  'ASIANPAINT': (3326.7893, 0.2437490583662306),
#  'AXISBANK': (976.059, 0.1240190798584342),
#  'BAJAJ-AUTO': (4663.6597, 0.0356006006006039),
#  'BAJAJFINSV': (1523.8195, 0.1820781696854217),
#  'BAJFINANCE': (7270.728, 0.302505242246991),
#  'BHARTIARTL': (829.71295, -0.1789039942252246),
#  'BPCL': (372.71188, 0.0703127936635841),
#  'BRITANNIA': (5056.1396, -0.006138694143112),
#  'CIPLA': (1012.9111, 0.1741680265044806),
#  'COALINDIA': (227.2668, 0.0073927392739226),
#  'DIVISLAB': (3573.9048, 0.2160506982221966),
#  'DRREDDY': (4912.423, 0.2289847384314036),
#  'EICHERMOT': (3552.38, -0.272872743606294),
#  'GRASIM': (1769.3655, 0.0150076309988243),
#  'HCLTECH': (1167.211, -0.123133530141623),
#  'HDFC': (2657.3677, -0.0444716104643657),
#  'HDFCBANK': (1610.6666, 0.1969891135303276),
#  'HDFCLIFE': (643.8163, 0.0180674227124297),
#  'HEROMOTOCO': (2794.282, -0.2291569964651575),
#  'HINDALCO': (429.0568, 0.0248980067606996),
#  'HINDUNILVR': (2680.8123, 0.1760883375060775),
#  'ICICIBANK': (926.4529, 0.0975528064394192),
#  'INDUSINDBK': (1301.8055, 0.262284349969181),
#  'INFY': (1303.0366, -0.027880926806827),
#  'ITC': (451.86023, -0.2185646461300698),
#  'JSWSTEEL': (771.61285, -0.1988165297807666),
#  'KOTAKBANK': (1847.1595, 0.1713394793926236),
#  'LT': (2380.76, -0.0520570948782444),
#  'MARUTI': (9501.587, 0.0999462711096508),
#  'M_M': (1397.0828, 0.0453149056536117),
#  'NESTLEIND': (23010.379, 0.2388485573205675),
#  'NTPC': (187.91376, 0.220671999999998),
#  'ONGC': (156.82765, -0.2685850556438756),
#  'POWERGRID': (248.23297, -0.2078512562814092),
#  'RELIANCE': (2552.582, -0.1766845254389749),
#  'SBILIFE': (1293.9965, -0.1044891342108264),
#  'SBIN': (565.70667, -0.2984367289390106),
#  'SUNPHARMA': (991.14087, -0.0815696355663184),
#  'TATACONSUM': (859.29706, 0.0462288974269426),
#  'TATAMOTORS': (1395.4204, -0.0737298148877563),
#  'TATASTEEL': (114.223015, -0.0236192560175022),
#  'TCS': (3223.4597, -0.1313721845276802),
#  'TECHM': (1109.495, 0.1937056937734214),
#  'TITAN': (2973.914, -0.0028917283120304),
#  'ULTRACEMCO': (8261.635, 0.2205993849662397),
#  'UPL': (685.5683, 0.3099422049893886),
#  'WIPRO': (382.4655, -0.009019607843132)}

# deposit = 30000

# sentimental_analysis(dict1, deposit)




