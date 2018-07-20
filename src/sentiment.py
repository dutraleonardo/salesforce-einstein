import json
import time
from language.sentiment.prediction import Predictions
from constants import ACCESS_TOKEN

text_list = []

def main():

    with open('static/input.csv', 'rU') as csvfile:
        reader = (line.replace('\r', '') for line in csvfile)
        for row in reader:
            text_list.append(row)

    access_token = ACCESS_TOKEN
    model_id = 'CommunitySentiment'
    prediction = Predictions(access_token=access_token)

    try:
        for document in text_list[1:]:
            response = prediction.predict_sentiment(document, model_id)
            probabilities = response['probabilities']
            print(document)
            print('#' * 50)
            for x in probabilities:
                print(str(x['label']) + ": " + str(x['probability']))
            print('#'*50)
            time.sleep(2)


    except TypeError:
        print('response ok? ' + str(response.ok))
        print('response content: ' + str(response.content))
    return True

if __name__ == "__main__":
    main()
