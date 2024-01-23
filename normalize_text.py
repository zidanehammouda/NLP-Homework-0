import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import string
import argparse
import matplotlib.pyplot as plt
import math

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def readfile(file):
  with open(file, 'r',encoding='utf-8') as file:
    return file.read()

def tokenize(text):
  tokens = text.split()
  return tokens

def lower(tokens):
  return [token.lower() for token in tokens]

def remove_punctuation(tokens):
  return [''.join(char for char in token if char not in string.punctuation) for token in tokens]

def remove_stopwords(tokens):
  stop_words = set(stopwords.words('english'))
  return [token for token in tokens if token.lower() not in stop_words]

def stem(tokens):
  stemmer = PorterStemmer()
  return [stemmer.stem(token.lower()) for token in tokens]

def get_wordnet_pos(tag):
      if tag.startswith('J'):
          return wordnet.ADJ
      elif tag.startswith('V'):
          return wordnet.VERB
      elif tag.startswith('N'):
          return wordnet.NOUN
      elif tag.startswith('R'):
          return wordnet.ADV
      else:
          return wordnet.NOUN

def lem(tokens):
  lemmatizer = WordNetLemmatizer()
  tagged = nltk.pos_tag(tokens)
  return [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in tagged]

def remove_numeric(tokens):
   return [token for token in tokens if not any(char.isdigit() for char in token)]

def normalize(text):
  tokens = tokenize(text)
  tokens =lower(tokens)
  tokens =remove_stopwords(tokens)
  tokens =remove_punctuation(tokens)
  tokens =remove_numeric(tokens)
  tokens =stem(tokens)
  tokens =lem(tokens)

  return tokens

def count(tokens):
  token_counts = {}
  for token in tokens:
    if token in token_counts:
        token_counts[token] += 1
    else:
        token_counts[token] = 1

  sorted_token_counts = sorted(token_counts.items(), key=lambda item: item[1], reverse=True)

  with open("output.txt", "w") as file:
        file.write(f"Number of unique tokens: {len(sorted_token_counts)}\n")
        
        file.write("\nMost used 10 words:\n")
        for i in range(10):
            if sorted_token_counts[i]:
                file.write(f"{sorted_token_counts[i][0]}: {sorted_token_counts[i][1]}\n")
            else:
                break
        
        file.write("\nLess used 10 words:\n")
        for i in range(len(sorted_token_counts)-1, len(sorted_token_counts)-11, -1):
            if sorted_token_counts[i]:
                file.write(f"{sorted_token_counts[i][0]}: {sorted_token_counts[i][1]}\n")
            else:
                break
            
  return sorted_token_counts

def visualize(sorted_tokens):
  words = [token[0] for token in sorted_tokens]
  frequencies = [token[1] for token in sorted_tokens]
  xticks = [int(round((i/10)*len(words),-2)) for i in range(1,11)]
  log_frequencies = [math.log10(token[1]) for token in sorted_tokens]
  plt.figure(figsize=(10, 5))
  plt.bar(range(1, len(words) + 1), log_frequencies, color='skyblue')
  plt.xlabel('Word Rank')
  plt.ylabel('Frequency')
  plt.title('Word Frequencies - Log Bar Chart')
  plt.xticks(xticks)
  plt.savefig('vis1.png', bbox_inches='tight')


  plt.figure(figsize=(10, 5))
  plt.plot(range(1, len(words) + 1), log_frequencies, color='skyblue')
  plt.xlabel('Word Rank')
  plt.ylabel('Frequency')
  plt.title('Word Frequencies - Log line Chart')
  plt.xticks(xticks)
  plt.savefig('vis2.png', bbox_inches='tight')

def process_file(filename, to_lower, to_stem, to_lem, to_rm_punct, to_rm_stpw,tp_rm_num):

  print("filename:",filename)
  print("Normalizations:",end=' ')
  for index,(arg_name, arg_value) in enumerate(locals().items()):
    if arg_value and index >0:
      print(arg_name,end=' ')
    
  print("")
  

  text = readfile(filename)
  tokens = tokenize(text)
  print("number of tokens before processing", len(tokens))

  if to_lower:
    tokens = lower(tokens)
    print("number of tokens after lower:", len(tokens))
  if to_rm_stpw:
    tokens = remove_stopwords(tokens)
    print("number of tokens after stpw:", len(tokens))
  if to_rm_punct:
    tokens = remove_punctuation(tokens)
    print("number of tokens after punct:", len(tokens))
  if tp_rm_num:
    tokens = remove_numeric(tokens)
    print("number of tokens after removing numericals:", len(tokens))
  if to_stem:
    tokens = stem(tokens)
    print("number of tokens after stem:", len(tokens))
  if to_lem:
    tokens = lem(tokens)
    print("number of tokens after lem:", len(tokens))
  
  print("")

  sorted_tokens = count(tokens)
  visualize(sorted_tokens)
  

def main():

  parser = argparse.ArgumentParser(description='Text file normalization script')

  parser.add_argument('filename', type=str, help='Path to the text file')

  parser.add_argument('--lower', action='store_true', help='Convert text to lowercase')
  parser.add_argument('--stem', action='store_true', help='Stem the words in the text')
  parser.add_argument('--lem', action='store_true', help='Lemmatize the words in the text')
  parser.add_argument('--punct', action='store_true', help='Remove punctuation from the text')
  parser.add_argument('--stopw', action='store_true', help='Remove stopwords from the text')
  parser.add_argument('--num', action='store_true', help='Remove stopwords from the text')


  args = parser.parse_args()

  process_file(args.filename, args.lower, args.stem, args.lem,args.punct,args.stopw,args.num)


if __name__ == "__main__":
    main()