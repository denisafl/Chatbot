import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random 
import json
import pickle
#part 1 ----------
#Loading all of our words and our labels
#Getting our documents ready with all of our patterns
with open("intents.json") as file:
    data = json.load(file)
    
print(data["intents"])

#looping through json file 
try: 
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f) 
except:
    words = [] #empty list
    labels = []
    
     # for each pattern put another element in docs_y list that stands for what intent it s a part of
    
    docs_x = [] # list of the different patterns
    docs_y = [] # tag for words
    
    # loop through all of dictionary in json file of form {tag , patterns....}
    for intent in data["intents"]: 
        for pattern in intent["patterns"]:
            #next line will transforme words like 'whats'->what ;  'help?'->help, in order to get the root of the word
            #get the word with nltk
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            #this way we will get all of the different tags that we have
            docs_y.append(intent["tag"])
            
            
        if intent["tag"] not in labels:
            labels.append(intent["tag"])
    
    ####---------Part 2
            
    #we are going to step all the words
    #   that we have in the words list
    #   and remove any duplicate element
    #   cause we want to figure out what 
    #   kind of vocabulary size of the 
    #   model is 
    #[how many words it has seen already]
    
    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
              #convert into lower case to not get mixed interpretation
    words = sorted(list(set(words)))
            #set -> make sure there is no duplicate
            #list->convert the set into a list datatype
            # sorted -> sort the final output
    labels = sorted(labels)
    
    #bags of words = one hot encoded
    #check if the word is there or not
    #CONVERT WORDS OF STRINGS TO LIST OF NUMBERS
    training = []
    output = []   
    
    out_empty = [0 for _ in range(len(labels))]
    
    for x, doc in enumerate(docs_x):
        bag = []
        
        #our bag is a list like [0,0,1,0..0,1]
        #each time a word occure change 0 to 1
        #increase every time a word is found
        wrds = [stemmer.stem(w) for w in doc]
        
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        
        training.append(bag)
        output.append(output_row)
        
    training = numpy.array(training)
    output = numpy.array(output)


######writting the model        

tensorflow.compat.v1.reset_default_graph()  

net = tflearn.input_data(shape=[None, len(training[0])])
#   define the input shape that we
#   expected from our model 
       
net = tflearn.fully_connected(net, 8)
    #connecting 8 neurons for that hidden data above
net = tflearn.fully_connected(net, 8)
    # so we connect 2 hidden data

net = tflearn.fully_connected(net, len(output[0]), activation = "softmax")
    #allow us to get probabilities
    #for each output
#   -
#   Softmax will give us a probability for each neuron in
#    this layer (" len(output[0])"))
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
        model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size = 8, show_metric = True)
    model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))

chat()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        