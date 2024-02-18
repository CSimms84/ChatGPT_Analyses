import json
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

with open('conversations.json', 'r') as file:
    conversations = json.load(file)

word_frequencies = Counter()

stop_words = set(stopwords.words('english'))
custom_exclusions = {'create', 'import', 'error', 'set', 'method', 'application', 'might', 'string',
                     'return', 'system', 'need', 'different', 'command', 'value', 'e', 'would', 'name', 'web', 'table', 'may',
                     'use', 'user', 'used', 'like', 'using', 'file', 'new', 'specific', 'code', 'data', 'example', 'function',
                     'element', 'type', 'object', 'elements', 'project', 'public', 'make', 'also', 'key', 'access', 'div', 'const',
                     'ensure', 'within', 'n', 'instance', 'version', 'td', 'configuration', 'email', 'files', 'provides', 'run',
                     'network', 'provide', 'directory', 'could', 'applications', 'information', 'process', 'based', 'check',
                     'list', 'model', 'one', 'content', 'text', 'running', 'environment', 'int', 'path', 'allows', 'input',
                     'time', 'number', 'performance', 'want', 'add', 'service', 'script', 'structure', 'systems', 'message',
                     'various', 'security', 'operations', 'typically', 'request', 'case', 'methods', 'null', 'including',
                     'development', 'software', 'components', 'provided', 'often', 'multiple', 'include','var','tr','r','bmak',
                     'explain','none','true','jan','err','void','private','p','sh','class','nbsp','info','else','c','color',
                     'span','val','start','center','l','line','br','password','static','try','get','false','u','b','top',
                     'array','failed','unit','let','see','result','default','catch','response','id','next'}
stop_words.update(custom_exclusions)

def process_text(text):
    words = word_tokenize(text)
    words_filtered = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    return words_filtered

for conversation in conversations:
    for message_id, message_details in conversation['mapping'].items():
        if message_details['message']:
            message = message_details['message']
            if message['author']['role'] == 'user' and message['content']['content_type'] == 'text':
                text = " ".join(message['content']['parts'])
                word_frequencies.update(process_text(text))

print("Most common words in user messages:")
for word, count in word_frequencies.most_common(20):
    print(f"{word}: {count}")
