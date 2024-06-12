#!/usr/bin/env python
# coding: utf-8

# # Mendefinisikan Variable Penting

# In[1]:


# Variable Penting
# ListLabelUnique = []
# ListLabelSpesifikUnique = []
testSize = 0.20 #Pembagian ukuran datatesing

MAX_NB_WORDS = 100000 #Maximum jumlah kata pada vocabulary yang akan dibuat
max_seq_len = 0 #Panjang kalimat maximum

num_epochs = 100 #Jumlah perulangan / epoch saat proses training


# # Import Module

# In[2]:


# Import CSV
import csv
# Import json
import json
# Import Pandas
import pandas as pd
# Settingan di Pandas untuk menghilangkan warning
pd.options.mode.chained_assignment = None  # default='warn'

# Import Numpy
import numpy as np

# Loading Bar TQDM
from tqdm import tqdm

# Stopword Removal
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize

# Stemming (Sastrawi)
# !pip install Sastrawi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Tokenizer
from keras.preprocessing.text import Tokenizer

# pad_sequences untuk google colab
# from keras.utils import pad_sequences
# pad_sequences untuk jupter-lab
from keras_preprocessing.sequence import pad_sequences

# Pickle FastText
import pickle

# Split Data
from sklearn.model_selection import train_test_split

# Label Encoder
from sklearn.preprocessing import LabelEncoder

# Model Building
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.backend import clear_session
from keras.models import load_model

# Callbacks
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

# Plot Model
from keras.utils import plot_model

# Grafik
import matplotlib.pyplot as plt
import seaborn as sn
sn.set_style("whitegrid")

# Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn


# # Function Membaca file JSON menjadi DataFrame

# In[3]:


def read_data(filename):
    # Membaca file intents.json untuk dijadikan Dataframe
    df = pd.read_csv(filename) 
    return df


# # Memanggil Function read_data

# In[4]:


# Memanggil Function Read Data
FileIntents = './Dataset/Intents.csv'
df = read_data(FileIntents)
df


# In[5]:


# Membuat fungsi label encoder
def encode_label(df):
    # Encoding Categorical Data (Mengubah data kategorikal menjadi angka)
    from sklearn.preprocessing import LabelEncoder
    LE = LabelEncoder()
    df['Label_Encoded'] = LE.fit_transform(df['Label'])
    return df


# In[6]:


# Menampilkan Label sebelum dan sesudah encoded
df = encode_label(df)
pd.set_option("max_rows", None)
label = df[['Label','Label_Encoded']].sort_values(["Label_Encoded"],ascending=[True]).drop_duplicates().reset_index(drop=True)
label


# In[7]:


# Shape dari Label_Spesifik_Encoded
pd.get_dummies(df['Label']).values.shape


# In[8]:


def preprocessing(data):
    # Case Folding
    data['lower'] = data['Pertanyaan'].str.lower()
    
    # Punctual Removal
    data['punctual'] = data['lower'].str.replace('[^a-zA-Z0-9]+',' ', regex=True)
    
    # Normalization
    kamus_baku = pd.read_csv('kata_baku.csv', sep=";")
    dict_kamus_baku = kamus_baku[['slang','baku']].to_dict('list')
    dict_kamus_baku = dict(zip(dict_kamus_baku['slang'], dict_kamus_baku['baku']))
    norm = []
    for i in data['punctual']:
        res = " ".join(dict_kamus_baku.get(x, x) for x in str(i).split())
        norm.append(str(res))
    data['normalize'] = norm
    
    # Stopword Removal
    stop_words = set(stopwords.words('indonesian'))
    swr = []
    from tqdm import tqdm
    for i in tqdm(data['normalize']):
        tokens = word_tokenize(i)
        filtered = [word for word in tokens if word not in stop_words]
        swr.append(" ".join(filtered))
    data['stopwords'] = swr
    
    # Stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stem = []
    from tqdm import tqdm
    for i in tqdm(data['stopwords']):
        stem.append(stemmer.stem(str(i)))
    data['stemmed'] = stem
    
    return data


# In[9]:


df = preprocessing(df)


# In[10]:


# data splitting
df_training, df_testing = train_test_split(df, test_size=testSize, random_state=42, shuffle=True)
df_training = df_training.reset_index(drop=True)
df_testing  = df_testing.reset_index(drop=True)


# In[11]:


# Mengecek panjang kalimat maksimum
longest_string = max(df['stemmed'].values.tolist(), key=len)
max_seq_len = len(longest_string.split())
pickle.dump(max_seq_len, open('max_seq_len.pkl','wb'))
print(longest_string)
print(max_seq_len)


# In[12]:


def tokenize_corpus(data_corpus):
    global corpus_tokenizer
    from keras.preprocessing.text import Tokenizer
    corpus_tokenizer = Tokenizer(oov_token="<OOV>")
    corpus_tokenizer.fit_on_texts(data_corpus['stemmed'])
    corpus_sequences = corpus_tokenizer.texts_to_sequences(data_corpus['stemmed'])
    corpus_word_index = corpus_tokenizer.word_index
    return corpus_sequences, corpus_word_index

def tokenize_training(data_training):
    global model_tokenizer #Menggunakan variabel global agar 'tokenizer' bisa dipake di luar fungsi ini
    from keras.preprocessing.text import Tokenizer
    model_tokenizer = Tokenizer(oov_token = "<OOV>")
    model_tokenizer.fit_on_texts(data_training['stemmed'])
    model_word_index = model_tokenizer.word_index
    train_sequences = model_tokenizer.texts_to_sequences(data_training['stemmed'])
    max_seq_len = pickle.load(open('max_seq_len.pkl', 'rb'))
    train_sequences_padded = pad_sequences(train_sequences, maxlen = max_seq_len)
    return train_sequences, train_sequences_padded, model_word_index

def tokenize_testing(data_testing):
    test_model_sequences = model_tokenizer.texts_to_sequences(data_testing['stemmed'])
    max_seq_len = pickle.load(open('max_seq_len.pkl', 'rb'))
    test_model_sequences_padded = pad_sequences(test_model_sequences, maxlen = max_seq_len)
    test_corpus_sequences = corpus_tokenizer.texts_to_sequences(data_testing['stemmed'])
    return test_model_sequences, test_model_sequences_padded, test_corpus_sequences


# In[13]:


corpus_sequences, corpus_word_index = tokenize_corpus(df)
train_sequences, train_sequences_padded, model_word_index  = tokenize_training(df_training)
test_model_sequences, test_model_sequences_padded, test_corpus_sequences = tokenize_testing(df_testing)
print("Sebelum Tokenizer :" , df_training['stemmed'][0])
print("Setelah Tokenizer :" , train_sequences_padded[0])


# In[14]:


# Alasan sequence model untuk predict berbentuk 2 dimensional array
print(corpus_tokenizer.texts_to_sequences(['idx30']))
print(corpus_tokenizer.texts_to_sequences('idx30'))


# In[15]:


print("Model Tokenized Word :",len(list(model_word_index.keys())))
print("Corpus Tokenized Word :",len(list(corpus_word_index.keys())))


# In[16]:


corpus_word_index


# In[17]:


pickle.dump(model_tokenizer, open('model_tokenizer.pkl','wb'))
pickle.dump(corpus_tokenizer, open('corpus_tokenizer.pkl','wb'))
pickle.dump(corpus_word_index, open('corpus_word_index.pkl','wb'))


# In[18]:


# pertanyaan df_training setelah di sequencing
df_training['Sequences'] = train_sequences


# In[19]:


# Pertanyaan df_testing setelah di tokenize
df_testing = df_testing.reset_index(drop=True) #Reset Index
df_testing['Model_Sequences'] = test_model_sequences
df_testing['Corpus_Sequences'] = test_corpus_sequences


# In[20]:


df['Sequences'] = corpus_sequences


# In[21]:


nb_words = min(MAX_NB_WORDS, len(model_word_index)+1)


# In[22]:


# Penentuan X (input) dan Y (output)
X_train = train_sequences_padded
X_test = test_model_sequences_padded

# One Hot Encoded Label
Y_train = pd.get_dummies(df_training['Label']).values
Y_test = pd.get_dummies(df_testing['Label']).values


# In[23]:


model = keras.Sequential([
        keras.layers.Embedding(nb_words,100,input_length=max_seq_len),
        keras.layers.LSTM(32),# returns a sequence of vectors of dimension 32 ini
        keras.layers.Dense(len(Y_train[0]), activation='softmax')
    ])
model.summary()


# In[24]:


# One Hot Label Encoded
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', patience=3)
chatbotmodel = model.fit(X_train, Y_train,
        epochs = 100,
        callbacks = [es],
        validation_split=0.15,
        verbose = True # Verbose = 0 (tidak nampak progress), 1/True (progress bar), 2 (angka)
)


# In[25]:


plot_model(model,to_file='model.png',show_shapes=True,show_layer_names=True)


# In[26]:


model.save('chatbotmodel.h5', chatbotmodel)


# In[27]:


plt.figure()
plt.plot(chatbotmodel.history['loss'], lw=2.0, color='b', label='train')
plt.plot(chatbotmodel.history['val_loss'], lw=2.0, color='r', label='val')
plt.title("Loss History")
plt.xlabel('Epochs')
plt.ylabel('Cross-Entropy Loss')
plt.legend(loc='upper right')
plt.show()


# In[28]:


plt.figure()
plt.plot(chatbotmodel.history['accuracy'], lw=2.0, color='b', label='train')
plt.plot(chatbotmodel.history['val_accuracy'], lw=2.0, color='r', label='val')
plt.title("Accuracy History")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()


# In[29]:


def predict_test(model,test_sequences_padded):
    categorical_predicted_label = []
    onehot_predicted_label = model.predict(test_sequences_padded)
    for i in range(0,len(test_sequences_padded)):
        categorical_predicted_label.append(onehot_predicted_label[i].argmax())
    return onehot_predicted_label, categorical_predicted_label


# In[30]:


pd.set_option("max_rows", None)
onehot_predicted_label, df_testing['Predicted_Label'] = predict_test(model,test_model_sequences_padded)
df_testing[['Pertanyaan','Label_Encoded','Predicted_Label']].loc[df_testing['Label_Encoded'] != df_testing['Predicted_Label']]


# In[31]:


def plot_confusion_matrix(data, labels, output_filename):
    import seaborn as sn
    sn.set(color_codes=True)
    import matplotlib.pyplot as plt
    plt.figure(1, figsize=(9, 6))
    plt.title("Confusion Matrix")
    sn.set(font_scale=1.4)
    ax = sn.heatmap(data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'})
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set(ylabel="True Label", xlabel="Predicted Label")
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close()
cm = confusion_matrix(df_testing[['Label_Encoded']], df_testing[['Predicted_Label']])
plot_confusion_matrix(cm, ['IPO','Investasi','Istilah','Strategi','Trading'], 'confusion_matrix.png')


# In[32]:


label


# In[33]:


df_testing[['Label_Encoded','Predicted_Label']].loc[df_testing['Label_Encoded'] != df_testing['Predicted_Label']].shape


# In[34]:


# Pembagian dataframe_corpus berdasarkan label untuk matching
df_IPO = df[['Pertanyaan','Jawaban','Sequences']].loc[df['Label_Encoded']==int(0)].reset_index(drop=True)
df_Investasi = df[['Pertanyaan','Jawaban','Sequences']].loc[df['Label_Encoded']==int(1)].reset_index(drop=True)
df_Istilah = df[['Pertanyaan','Jawaban','Sequences']].loc[df['Label_Encoded']==int(2)].reset_index(drop=True)
df_Strategi = df[['Pertanyaan','Jawaban','Sequences']].loc[df['Label_Encoded']==int(3)].reset_index(drop=True)
df_Trading = df[['Pertanyaan','Jawaban','Sequences']].loc[df['Label_Encoded']==int(4)].reset_index(drop=True)


# In[35]:


pickle.dump(df_IPO, open('df_IPO.pkl','wb'))
pickle.dump(df_Investasi, open('df_Investasi.pkl','wb'))
pickle.dump(df_Istilah, open('df_Istilah.pkl','wb'))
pickle.dump(df_Strategi, open('df_Strategi.pkl','wb'))
pickle.dump(df_Trading, open('df_Trading.pkl','wb'))


# In[36]:


# Pembagian dataframe_testing berdasarkan predicted label untuk matching
def matching_testing(df_testing):
    Prediksi_Jawaban = []
    for row in range(len(df_testing)):
        #Mengambil Predicted_Label per baris
        Predicted_Label = df_testing['Predicted_Label'].iloc[row] 
        # Menentukan df yang dipakai untuk matching
        if(Predicted_Label==0):
            df_IPO = pickle.load(open('df_IPO.pkl', 'rb'))
            check_df = df_IPO
        elif(Predicted_Label==1):
            df_Investasi = pickle.load(open('df_Investasi.pkl', 'rb'))
            check_df = df_Investasi
        elif(Predicted_Label==2):
            df_Istilah = pickle.load(open('df_Istilah.pkl', 'rb'))
            check_df = df_Istilah
        elif(Predicted_Label==3):
            df_Strategi = pickle.load(open('df_Strategi.pkl', 'rb'))
            check_df = df_Strategi
        elif(Predicted_Label==4):
            df_Trading = pickle.load(open('df_Trading.pkl', 'rb'))
            check_df = df_Trading
    
        Compatibility = [0]*len(check_df)
        
        # Looping tiap baris Sequences check_df
        index = 0
        for check_sequences in check_df['Sequences']:
            # Looping tiap element Sequences df testing per baris
            for element in df_testing['Corpus_Sequences'].iloc[row]:
                if(element in check_sequences):
                    Compatibility[index]+=1
            Compatibility[index] = Compatibility[index]/len(df_testing['Corpus_Sequences'].iloc[row])
            index += 1
    
        index_max_compatibility = []
        for idx, value in enumerate(Compatibility):
            if value == max(Compatibility):
                index_max_compatibility.append(idx)

        perfect_compatibilty_sequence = []
        perfect_compatibilty_index = 0
        for idx in index_max_compatibility:
            if (idx == index_max_compatibility[0]):
                perfect_compatibilty_sequence = check_df['Sequences'].iloc[idx]
                perfect_compatibilty_index = idx
            else:
                if(len(check_df['Sequences'].iloc[idx]) <= len(perfect_compatibilty_sequence)):
                    perfect_compatibilty_sequence = check_df['Sequences'].iloc[idx]
                    perfect_compatibilty_index = idx

        Prediksi_Jawaban.append(check_df['Jawaban'].iloc[perfect_compatibilty_index])
    return Prediksi_Jawaban


# In[37]:


Prediksi_Jawaban = matching_testing(df_testing)
df_testing['Prediksi_Jawaban'] = Prediksi_Jawaban


# In[38]:


df_testing[['Pertanyaan','Jawaban','Prediksi_Jawaban']].loc[df_testing['Jawaban']!=df_testing['Prediksi_Jawaban']]


# In[39]:


df_testing[['Pertanyaan','Jawaban','Prediksi_Jawaban']]

