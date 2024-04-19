# Chatbot
This project was made using python

Based on our code, we first load words and labels, then get the documents ready with all the patterns.

The next step is to loop through json file using a try – except method. We created empty lists for words and labels and for each pattern we put another element in docs_y list that stands for what intent it is a part of.
Using a for loop, we go through all of dictionary in json file of form {tag , patterns....}.
For part 2, we are going to step all the words that we have in the words list and remove any duplicate element cause we want to figure out what  kind of vocabulary size of the model is.
Next, we check if the word is there or not and convert words of strings to list of numbers. Each time a word occure, change 0 to 1 (increase every time a word is find).
For the model we used regression.
Then, we define a function bag_of_words which takes 2 parameters and tokenise s, then enumerate the words.
And the function chat makes the prediction for each input and prints random answers.
