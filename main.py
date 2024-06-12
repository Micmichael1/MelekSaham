# Dataframe Module
import pandas as pd

# Stopword Removal Module
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Tokenizer Module
from nltk.tokenize import word_tokenize

# Stemmer Module
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Tokenizer Module
# from keras.preprocessing.text import Tokenizer

# Padding Module
from keras_preprocessing.sequence import pad_sequences

# Import/Export Object Module
import pickle

# load_model Module
from keras.models import load_model

# fuzzywuzzy Module (Untuk Function Fuzzy_Recommend)
from fuzzywuzzy import process, fuzz

# pertanyaan_oov Module
from csv import writer

# Connect to Telegram Module
from aiogram import Bot, Dispatcher, executor, types

# Load json file Module
import json

# Module untuk penarikan API
import requests

# Module untuk menghitung QA execution time
import time

class OOV():

    def Preprocessing(self, data):
        # Case Folding
        data['lower'] = data['Pertanyaan'].str.lower()

        # Punctual Removal
        data['punctual'] = data['lower'].str.replace('[^a-zA-Z0-9]+', ' ', regex=True)

        # Normalization
        kamus_baku = pd.read_csv('kata_baku.csv', sep=";")
        dict_kamus_baku = kamus_baku[['slang', 'baku']].to_dict('list')
        dict_kamus_baku = dict(zip(dict_kamus_baku['slang'], dict_kamus_baku['baku']))
        norm = []
        for i in data['punctual']:
            res = " ".join(dict_kamus_baku.get(x, x) for x in str(i).split())
            norm.append(str(res))
        data['normalize'] = norm

        # Stopword Removal
        stop_words = set(stopwords.words('indonesian'))
        swr = []
        for i in data['normalize']:
            tokens = word_tokenize(i)
            filtered = [word for word in tokens if word not in stop_words]
            swr.append(" ".join(filtered))
        data['stopwords'] = swr

        # Stemming
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        stem = []
        for i in data['stopwords']:
            stem.append(stemmer.stem(str(i)))
        data['stemmed'] = stem

        return data

    def Model_Sequencing(self, data):
        model_tokenizer = pickle.load(open('model_tokenizer.pkl', 'rb'))
        model_sequences = model_tokenizer.texts_to_sequences(data['stemmed'])
        max_seq_len = pickle.load(open('max_seq_len.pkl', 'rb'))
        model_sequences_padded = pad_sequences(model_sequences, maxlen=max_seq_len)
        return model_sequences_padded

    def Predict_Label(self, model_sequences_padded):
        model = load_model('chatbotmodel.h5')
        categorical_predicted_label = []
        onehot_predicted_label = model.predict(model_sequences_padded)
        for i in range(0, len(model_sequences_padded)):
            categorical_predicted_label.append(onehot_predicted_label[i].argmax())
        return categorical_predicted_label

    def Menambahkan_Pertanyaan_OOV(self):
        try :
            data = pd.read_csv('pertanyaan_oov.csv', header=None)
            # print(data.shape)
            if (data.shape[1]==1):
                print("Terdapat Pertanyaan OOV yang belum dijawab")
            elif (data.shape[1]>1):
                print("Terdapat Pertanyaan OOV yang sudah dijawab")
                print ("Ditemukan",data.shape[0],"baris pertanyaan OOV yang dapat ditambahkan ke corpus")
                print("Proses Penambahan Pertanyaan OOV dimulai")
                data.columns = ['Pertanyaan', 'Jawaban']
                print("Tahap Preprocessing Pertanyaan OOV dimulai")
                data = self.Preprocessing(data)
                print("Tahap Preprocessing Pertanyaan OOV berhasil")
                print("Tahap Model Sequencing Pertanyaan OOV dimulai")
                model_sequences_padded = self.Model_Sequencing(data)
                print("Tahap Model Sequencing Pertanyaan OOV berhasil")
                print("Tahap Predict Label Pertanyaan OOV dimulai")
                data['Label'] = self.Predict_Label(model_sequences_padded)
                for i in range(len(data['Label'])):
                    label = data.loc[i,'Label']
                    if (label == 0):
                        data.loc[i,'Label'] = 'IPO'
                    elif (label == 1):
                        data.loc[i,'Label'] = 'Investasi'
                    elif (label == 2):
                        data.loc[i,'Label'] = 'Istilah'
                    elif (label == 3):
                        data.loc[i,'Label'] = 'Strategi'
                    elif (label == 4):
                        data.loc[i,'Label'] = 'Trading'
                print("Tahap Predict Label Pertanyaan OOV berhasil")
                data = data[['Pertanyaan','Jawaban','Label']]
                print("Tahap penambahan Pertanyaan OOV kedalam Corpus dimulai")
                intents = pd.read_csv('./Dataset/Intents.csv')
                intents = pd.concat([intents,data])
                intents.to_csv("./Dataset/Intents.csv",index=False)
                print("Tahap penambahan Pertanyaan OOV kedalam Corpus berhasil")
                print("Tahap pelatihan ulang model dimulai")
                exec(open("./main_notebook.py").read())
                print("Tahap pelatihan ulang model berhasil")
                print("Tahap pembersihan csv pertanyaan_oov.csv dimulai")
                f = open("pertanyaan_oov.csv", "w")
                f.truncate()
                f.close()
                print("Tahap pembersihan csv pertanyaan_oov.csv berhasil")
                print("Proses Penambahan Pertanyaan OOV selesai")
        except pd.errors.EmptyDataError:
            print("Tidak terdapat Pertanyaan OOV yang perlu ditambahkan")


class QA():

    def Preprocessing(self,input):
        print("Tahap Preprocessing Dimulai")

        # Case Folding
        data = pd.DataFrame([input], columns=['pertanyaan'])
        data['lower'] = data['pertanyaan'].str.lower()
        print("Tahap Case Folding Berhasil :",data['lower'].iloc[0])

        # Punctual Removal
        data['punctual'] = data['lower'].str.replace('[^a-zA-Z0-9]+', ' ', regex=True)
        print("Tahap Punctual Removal Berhasil :",data['punctual'].iloc[0])

        # Normalization
        kamus_baku = pd.read_csv('kata_baku.csv', sep=";")
        dict_kamus_baku = kamus_baku[['slang', 'baku']].to_dict('list')
        dict_kamus_baku = dict(zip(dict_kamus_baku['slang'], dict_kamus_baku['baku']))
        norm = []
        for i in data['punctual']:
            res = " ".join(dict_kamus_baku.get(x, x) for x in str(i).split())
            norm.append(str(res))
        data['normalize'] = norm
        print("Tahap Normalisasi Berhasil :",data['normalize'].iloc[0])

        # Stopword Removal
        stop_words = set(stopwords.words('indonesian'))
        swr = []
        for i in data['normalize']:
            tokens = word_tokenize(i)
            filtered = [word for word in tokens if word not in stop_words]
            swr.append(" ".join(filtered))
        data['stopwords'] = swr
        print("Tahap Stopword Removal Berhasil :",data['stopwords'].iloc[0])

        # Stemming
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        stem = []
        for i in data['stopwords']:
            stem.append(stemmer.stem(str(i)))
        data['stemmed'] = stem
        print("Tahap Stemming Berhasil :",data['stemmed'].iloc[0])
        return data

    def Find_oov(self,list_to_check, item_to_find):
        oov_index = []
        for idx, value in enumerate(list_to_check):
            if value == item_to_find:
                oov_index.append(idx)
        return oov_index

    def OOV_Checking(self,data):
        corpus_tokenizer = pickle.load(open('corpus_tokenizer.pkl', 'rb'))
        corpus_sequence = corpus_tokenizer.texts_to_sequences(data['stemmed'])
        oov_index = self.Find_oov(corpus_sequence[0],1)
        oov_words = []
        input_splitted = data['stemmed'].iloc[0].split()
        for i in oov_index:
            oov_words.append(input_splitted[i])
        return oov_words

    def Fuzzy_Recommend(self,oov_words,input):
        print("Tahap Fuzzy Recommend Dimulai")
        corpus_word_index = pickle.load(open('corpus_word_index.pkl', 'rb'))
        list_corpus_word_index = list(corpus_word_index.keys())
        recommended_input = []
        for oov_word in oov_words:
            recommended_input.append(process.extract(oov_word, list_corpus_word_index, scorer=fuzz.token_sort_ratio)[0][0])
            print(process.extract(oov_word, list_corpus_word_index, scorer=fuzz.token_sort_ratio))
        with open('pertanyaan_oov.csv', 'a') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow([input])
            f_object.close()
        return recommended_input

    def Add_To_Pertanyaan_OOV(self,input):
        print("Pertanyaan Yang Diberikan Tidak Terdapat Dalam Corpus")
        with open('pertanyaan_oov.csv', 'a') as f_object:
            print("Proses Memasukkan Pertanyaan Kedalam Pertanyaan_OOV.csv Dimulai")
            writer_object = writer(f_object)
            writer_object.writerow([input])
            f_object.close()
            print("Proses Memasukkan Pertanyaan Kedalam Pertanyaan_OOV.csv Selesai")
        return 0

    def Model_Sequencing(self,data):
        model_tokenizer = pickle.load(open('model_tokenizer.pkl', 'rb'))
        model_sequence = model_tokenizer.texts_to_sequences(data['stemmed'])
        max_seq_len = pickle.load(open('max_seq_len.pkl', 'rb'))
        model_sequence_padded = pad_sequences(model_sequence, maxlen=max_seq_len) #maxlen mengikuti maxlen model training
        print("Tahap Model Sequencing Berhasil :", model_sequence_padded)
        return model_sequence_padded

    def Predict_Label(self,sequence):
        model = load_model('chatbotmodel.h5')
        label = int(model.predict(sequence).argmax())
        if (label == 0):
            type_label = 'IPO'
        elif (label == 1):
            type_label = 'Investasi'
        elif (label == 2):
            type_label = 'Istilah'
        elif (label == 3):
            type_label = 'Strategi'
        elif (label == 4):
            type_label = 'Trading'
        print("Tahap Predict Label Berhasil :",type_label)
        return label

    def Corpus_Sequencing(self,data):
        corpus_tokenizer = pickle.load(open('corpus_tokenizer.pkl', 'rb'))
        corpus_sequence = corpus_tokenizer.texts_to_sequences(data['stemmed'])
        print("Tahap Corpus Sequencing Berhasil :",corpus_sequence)
        return corpus_sequence

    def Matching(self,sequence,predicted_label):
        print("Tahap Matching Dimulai")
        print("Sequence Pertanyaan: ",sequence)
        df_Investasi = pickle.load(open('df_Investasi.pkl', 'rb'))
        df_IPO = pickle.load(open('df_IPO.pkl', 'rb'))
        df_Istilah = pickle.load(open('df_Istilah.pkl', 'rb'))
        df_Strategi = pickle.load(open('df_Strategi.pkl', 'rb'))
        df_Trading = pickle.load(open('df_Trading.pkl', 'rb'))
        if (predicted_label == 0):
            check_df = df_IPO
        elif (predicted_label == 1):
            check_df = df_Investasi
        elif (predicted_label == 2):
            check_df = df_Istilah
        elif (predicted_label == 3):
            check_df = df_Strategi
        elif (predicted_label == 4):
            check_df = df_Trading

        Compatibility = [0] * len(check_df)
        # print("sudah sampai sini 2")
        # Looping tiap baris Sequences df corpus pilihan
        index = 0
        for check_sequences in check_df['Sequences']:
            # Looping tiap element Sequences df testing per baris
            for element in sequence[0]:
                # print(element)
                if (element in check_sequences):
                    # print("didalam if")
                    Compatibility[index] += 1
            #                     print(Compatibility)
            # Compatibility[index] = Compatibility[index] / len(check_sequences)
            Compatibility[index] = Compatibility[index] / len(sequence[0])
            index += 1
        print("Tahap Pengecekan Compatibility Berhasil")
        print("Hasil Pengecekan Compatibilty :",Compatibility)

        # Jika presentase compatibility sama maka diambil jawaban yang len corpus_sequence-nya terpanjang
        # Dimana hal ini menandakan kecocokan yang lebih menyeluruh
        # Penggunaan teknik ini untuk mencegah input panjang tetapi ada kata yang terdapat pada pertanyaan corpus yang pendek
        # Sehingga menghasilkan Compatibilty tinggi misal input apa itu saham preferen? [2,3] terhadap corpus apa itu saham? [2] dan apa itu preferen? [3]
        print("Tahap Max Compatibility Dimulai")
        index_max_compatibility = []
        for idx, value in enumerate(Compatibility):
            if value == max(Compatibility):
                index_max_compatibility.append(idx)

        print("Index yang memiliki Max Compatibility :",index_max_compatibility)

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

        # Prediksi_Jawaban = check_df['Jawaban'].loc[Compatibility.index(max(Compatibility))]
        Prediksi_Jawaban = check_df['Jawaban'].iloc[perfect_compatibilty_index]
        print("Tahap Pengambilan Compatibility Maksimum Berhasil")
        print("Index dengan Compatibility Maksimum yang akan diambil adalah index ke :",perfect_compatibilty_index)
        return Prediksi_Jawaban

    def Handle_Response(self,input):
        data = self.Preprocessing(input)
        oov_words = self.OOV_Checking(data)
        if(len(oov_words)==0):
            print("Pertanyaan yang diberikan adalah :",input)
            model_sequence_padded = self.Model_Sequencing(data)
            predicted_label = self.Predict_Label(model_sequence_padded)
            corpus_sequence = self.Corpus_Sequencing(data)
            prediksi_jawaban = self.Matching(corpus_sequence,predicted_label)
            print("Tahap Pencarian Jawaban Berhasil")
            print("Jawaban atas pertanyaan '"+input+"' adalah '"+prediksi_jawaban+"'")
            return prediksi_jawaban
        else:
            Jawaban = "Maaf, saat ini chatbot masih tidak memiliki pengetahuan mengenai '" + ' '.join(oov_words) + "'\n" + "Jawaban atas pertanyaan tersebut akan segera ditambahkan."
            self.Add_To_Pertanyaan_OOV(input)
            # recommended_input = self.Fuzzy_Recommend(oov_words, input)
            # Jawaban = Jawaban + "Apakah yang anda maksud adalah '" + ' '.join(recommended_input) + "'?"
            # Jawaban = "Maaf, saya tidak memiliki pengetahuan mengenai "+ x for x in (data['stemmed'].iloc[0].split())
            return Jawaban

Token = '6150390103:AAGAYSgspKsA5V7ap8zAJQKezwddEVnSpog'
BOT_USERNAME = '@MelekSaham_bot'
bot = Bot(token=Token)
# bot = Bot(token=Token, proxy='http://proxy.server:3128')
dp = Dispatcher(bot)

@dp.message_handler(commands=['start', 'help'])
async def start(message):
    await message.reply('Selamat datang di MelekSaham!\n'
                        'Saya adalah sebuah Chatbot Edukasi Saham Indonesia.\n'
                        'Saya siap membantu anda mempelajari dunia investasi saham Indonesia.\n'
                        'Saya dilengkapi dengan modul pembelajaran yang mendalam, data-data saham, serta saya dapat menjawab pertanyaan seputar saham yang diberikan.\n'
                        'Tujuan saya adalah untuk membantu menjadikan anda investor yang percaya diri dan terinformasi.\n'
                        'Mari mulai eksplorasi saham Indonesia!\n\n'
                        'Gunakan perintah /list_perintah untuk mengakses semua perintah yang dapat digunakan')

@dp.message_handler(commands=['list_perintah'])
async def list_perintah(message):
    await message.reply('Perintah-perintah yang dapat digunakan yakni :\n'
                        '/start - Perintah untuk menampilkan deskripsi chatbot\n'
                        '/list_perintah - Perintah untuk menampilkan list jenis perintah yang dapat dijalankan\n'
                        '/moduls - Perintah untuk mengakses modul pembelajaran saham\n'
                        '/saham_per_sektor - Perintah untuk mengakses sektor beserta sahamnya yang terdaftar di BEI\n'
                        '/info - Perintah untuk mengakses info perusahaan\n'
                        '/harga - Perintah untuk mengakses harga saham\n'
                        '/tanya - Perintah untuk menanyakan langsung pertanyaan mengenai saham')

@dp.message_handler(commands=['moduls'])
async def moduls(message):
    await message.reply('Selamat datang di Modul Pembelajaran Saham!\n'
                        'Berikut merupakan modul-modul pembelajaran saham yang tersedia:\n'
                        'Modul 1. /modul_basic_pasar_modal - Modul ini berisi pemahaman awal mengenai pasar modal saham\n'
                        'Modul 2. /modul_membaca_laporan_keuangan - Modul ini berisi tentang cara membaca laporan keuangan perusahaan\n'
                        'Modul 3. /modul_analisis_fundamental - Modul ini berisi tentang cara menganalisis saham dari segi fundamental kinerja perusahaan\n'
                        'Modul 4. /modul_analisis_teknikal - Modul ini berisi tentang cara menganalisis saham dari segi pergerakan harga dan volume lampau\n'
                        'Modul 5. /modul_aksi_korporasi - Modul ini berisi tentang jenis-jenis aksi korporasi yang ada di pasar modal saham')

@dp.message_handler(commands=['modul_basic_pasar_modal'])
async def modul_basic_pasar_modal(message):
    await message.reply('Modul 1. Basic Pasar Modal'
                        'Berikut merupakan chapter-chapter yang tersedia:\n'
                        'Ch 1.1. /investasi_dan_kebebasan_finansial\n'
                        'Ch 1.2. /pasar_modal\n'
                        'Ch 1.3. /saham\n'
                        'Ch 1.4. /menyaring_saham\n'
                        'Ch 1.5. /prinsip_investor')

@dp.message_handler(commands=['investasi_dan_kebebasan_finansial'])
async def investasi_dan_kebebasan_finansial(message):
    await message.reply(json_modul['modul']['1']['1.1']+"\n\nMelanjutkan ke Ch 1.2. /pasar_modal")

@dp.message_handler(commands=['pasar_modal'])
async def investasi_dan_kebebasan_finansial(message):
    await message.reply(json_modul['modul']['1']['1.2']+"\n\nMelanjutkan ke Ch 1.3. /saham")

@dp.message_handler(commands=['saham'])
async def investasi_dan_kebebasan_finansial(message):
    await message.reply(json_modul['modul']['1']['1.3']+"\n\nMelanjutkan ke Ch 1.4. /menyaring_saham")

@dp.message_handler(commands=['menyaring_saham'])
async def investasi_dan_kebebasan_finansial(message):
    await message.reply(json_modul['modul']['1']['1.4']+"\n\nMelanjutkan ke Ch 1.5. /prinsip_investor")

@dp.message_handler(commands=['prinsip_investor'])
async def investasi_dan_kebebasan_finansial(message):
    await message.reply(json_modul['modul']['1']['1.5']+"\n\nKembali ke list modul /moduls")

@dp.message_handler(commands=['modul_membaca_laporan_keuangan'])
async def modul_basic_pasar_modal(message):
    await message.reply('Modul 2. Membaca Laporan Keuangan'
                        'Berikut merupakan chapter-chapter yang tersedia:\n'
                        'Ch 2.1. /memahami_laporan_keuangan\n'
                        'Ch 2.2. /laporan_posisi_keuangan\n'
                        'Ch 2.3. /laporan_laba_rugi\n'
                        'Ch 2.4. /laporan_arus_kas\n'
                        'Ch 2.5. /rasio_laporan_keuangan')

@dp.message_handler(commands=['memahami_laporan_keuangan'])
async def investasi_dan_kebebasan_finansial(message):
    await message.reply(json_modul['modul']['2']['2.1']+"\n\nMelanjutkan ke Ch 2.2. /laporan_posisi_keuangan")

@dp.message_handler(commands=['laporan_posisi_keuangan'])
async def investasi_dan_kebebasan_finansial(message):
    await message.reply(json_modul['modul']['2']['2.2']+"\n\nMelanjutkan ke Ch 2.3. /laporan_laba_rugi")

@dp.message_handler(commands=['laporan_laba_rugi'])
async def investasi_dan_kebebasan_finansial(message):
    await message.reply(json_modul['modul']['2']['2.3']+"\n\nMelanjutkan ke Ch 2.4. /laporan_arus_kas")

@dp.message_handler(commands=['laporan_arus_kas'])
async def investasi_dan_kebebasan_finansial(message):
    await message.reply(json_modul['modul']['2']['2.4']+"\n\nMelanjutkan ke Ch 2.5. /rasio_laporan_keuangan")

@dp.message_handler(commands=['rasio_laporan_keuangan'])
async def investasi_dan_kebebasan_finansial(message):
    await message.reply(json_modul['modul']['2']['2.5']+"\n\nKembali ke list modul /moduls")

@dp.message_handler(commands=['modul_analisis_fundamental'])
async def modul_basic_pasar_modal(message):
    await message.reply('Modul 3. Analisis Fundamental'
                        'Berikut merupakan chapter-chapter yang tersedia:\n'
                        'Ch 3.1. /memahami_fundamental_analisis\n'
                        'Ch 3.2. /analisis_kualitatif\n'
                        'Ch 3.3. /analisis_kuantitatif_1\n'
                        'Ch 3.4. /analisis_kuantitatif_2\n'
                        'Ch 3.5. /valuasi\n'
                        'Ch 3.6. /menghitung_standard_deviasi_saham')

@dp.message_handler(commands=['memahami_fundamental_analisis'])
async def investasi_dan_kebebasan_finansial(message):
    await message.reply(json_modul['modul']['3']['3.1']+"\n\nMelanjutkan ke Ch 3.2. /analisis_kualitatif")

@dp.message_handler(commands=['analisis_kualitatif'])
async def investasi_dan_kebebasan_finansial(message):
    await message.reply(json_modul['modul']['3']['3.2']+"\n\nMelanjutkan ke Ch 3.3. /analisis_kuantitatif_1")

@dp.message_handler(commands=['analisis_kuantitatif_1'])
async def investasi_dan_kebebasan_finansial(message):
    await message.reply(json_modul['modul']['3']['3.3']+"\n\nMelanjutkan ke Ch 3.4. /analisis_kuantitatif_2")

@dp.message_handler(commands=['analisis_kuantitatif_2'])
async def investasi_dan_kebebasan_finansial(message):
    await message.reply(json_modul['modul']['3']['3.4']+"\n\nMelanjutkan ke Ch 3.5. /valuasi")

@dp.message_handler(commands=['valuasi'])
async def investasi_dan_kebebasan_finansial(message):
    await message.reply(json_modul['modul']['3']['3.5']+"\n\nMelanjutkan ke Ch 3.6. /menghitung_standard_deviasi_saham")

@dp.message_handler(commands=['menghitung_standard_deviasi_saham'])
async def investasi_dan_kebebasan_finansial(message):
    await message.reply(json_modul['modul']['3']['3.6']+"\n\nKembali ke list modul /moduls")

@dp.message_handler(commands=['modul_analisis_teknikal'])
async def modul_basic_pasar_modal(message):
    await message.reply('Modul 4. Analisis Teknikal'
                        'Berikut merupakan chapter-chapter yang tersedia:\n'
                        'Ch 4.1. /memahami_teknikal_analisis\n'
                        'Ch 4.2. /membaca_grafik\n'
                        'Ch 4.3. /support_dan_resistance\n'
                        'Ch 4.4. /trend\n'
                        'Ch 4.5. /chart_pattern\n'
                        'Ch 4.6. /taktik_trading\n'
                        'Ch 4.7. /money_management')

@dp.message_handler(commands=['memahami_teknikal_analisis'])
async def memahami_teknikal_analisis(message):
    await message.reply(json_modul['modul']['4']['4.1']+"\n\nMelanjutkan ke Ch 4.2. /membaca_grafik")


@dp.message_handler(commands=['membaca_grafik'])
async def membaca_grafik(message):
    await message.reply(json_modul['modul']['4']['4.2']+"\n\nMelanjutkan ke Ch 4.3. /support_dan_resistance")
    await bot.send_photo(chat_id=message.chat.id, photo=open('./Dataset/gambar_modul/anatomi_candlestick.jpg', 'rb'))
    await bot.send_photo(chat_id=message.chat.id, photo=open('./Dataset/gambar_modul/OHLC_candlestick.jpg', 'rb'))
    await bot.send_photo(chat_id=message.chat.id, photo=open('./Dataset/gambar_modul/bullish_candlestick.jpg', 'rb'))
    await bot.send_photo(chat_id=message.chat.id, photo=open('./Dataset/gambar_modul/bearish_candlestick.jpg', 'rb'))
    await bot.send_photo(chat_id=message.chat.id, photo=open('./Dataset/gambar_modul/praktek_candlestick.jpg', 'rb'))
    await bot.send_photo(chat_id=message.chat.id, photo=open('./Dataset/gambar_modul/penerapan_candlestick.jpg', 'rb'))

@dp.message_handler(commands=['support_dan_resistance'])
async def support_dan_resistance(message):
    await message.reply(json_modul['modul']['4']['4.3']+"\n\nMelanjutkan ke Ch 4.4. /trend")
    await bot.send_photo(chat_id=message.chat.id, photo=open('./Dataset/gambar_modul/support_dan_resistance.jpg', 'rb'))
    await bot.send_photo(chat_id=message.chat.id, photo=open('./Dataset/gambar_modul/breakout_dan_breakdown.jpg', 'rb'))

@dp.message_handler(commands=['trend'])
async def trend(message):
    await message.reply(json_modul['modul']['4']['4.4']+"\n\nMelanjutkan ke Ch 4.5. /chart_pattern")
    await bot.send_photo(chat_id=message.chat.id, photo=open('./Dataset/gambar_modul/uptrend.jpg', 'rb'))
    await bot.send_photo(chat_id=message.chat.id, photo=open('./Dataset/gambar_modul/identifikasi_uptrend.jpg', 'rb'))
    await bot.send_photo(chat_id=message.chat.id, photo=open('./Dataset/gambar_modul/downtrend.jpg', 'rb'))
    await bot.send_photo(chat_id=message.chat.id, photo=open('./Dataset/gambar_modul/identifikasi_downtrend.jpg', 'rb'))
    await bot.send_photo(chat_id=message.chat.id, photo=open('./Dataset/gambar_modul/sideways.jpg', 'rb'))

@dp.message_handler(commands=['chart_pattern'])
async def chart_pattern(message):
    await message.reply(json_modul['modul']['4']['4.5']+"\n\nMelanjutkan ke Ch 4.6. /taktik_trading")
    await bot.send_photo(chat_id=message.chat.id, photo=open('./Dataset/gambar_modul/bearish_pattern.jpg', 'rb'))
    await bot.send_photo(chat_id=message.chat.id, photo=open('./Dataset/gambar_modul/bullish_pattern.jpg', 'rb'))

@dp.message_handler(commands=['taktik_trading'])
async def taktik_trading(message):
    await message.reply(json_modul['modul']['4']['4.6'] + "\n\nMelanjutkan ke Ch 4.7. /money_management")
    await bot.send_photo(chat_id=message.chat.id, photo=open('./Dataset/gambar_modul/buy_on_breakout.jpg', 'rb'))
    await bot.send_photo(chat_id=message.chat.id, photo=open('./Dataset/gambar_modul/buy_on_weakness.jpg', 'rb'))
    await bot.send_photo(chat_id=message.chat.id, photo=open('./Dataset/gambar_modul/sell_on_breakout.jpg', 'rb'))
    await bot.send_photo(chat_id=message.chat.id, photo=open('./Dataset/gambar_modul/sell_on_strength.jpg', 'rb'))

@dp.message_handler(commands=['money_management'])
async def money_management(message):
    await message.reply(json_modul['modul']['4']['4.7']+"\n\nKembali ke list modul /moduls")
    await bot.send_photo(chat_id=message.chat.id, photo=open('./Dataset/gambar_modul/risk_to_reward_ratio.jpg', 'rb'))
    await bot.send_photo(chat_id=message.chat.id, photo=open('./Dataset/gambar_modul/position_sizing.jpg', 'rb'))

@dp.message_handler(commands=['modul_aksi_korporasi'])
async def modul_basic_pasar_modal(message):
    await message.reply('Modul 5. Aksi Korporasi'
                        'Berikut merupakan chapter-chapter yang tersedia:\n'
                        'Ch 5.1. /memahami_aksi_korporasi\n'
                        'Ch 5.2. /IPO\n'
                        'Ch 5.3. /tutorial_pembelian_IPO\n'
                        'Ch 5.4. /dividen\n'
                        'Ch 5.5. /stock_split\n'
                        'Ch 5.6. /right_issue\n'
                        'Ch 5.7. /tender_offer')

@dp.message_handler(commands=['memahami_aksi_korporasi'])
async def memahami_teknikal_analisis(message):
    await message.reply(json_modul['modul']['5']['5.1']+"\n\nMelanjutkan ke Ch 5.2. /IPO")

@dp.message_handler(commands=['IPO'])
async def membaca_grafik(message):
    await message.reply(json_modul['modul']['5']['5.2']+"\n\nMelanjutkan ke Ch 5.3. /tutorial_pembelian_IPO")

@dp.message_handler(commands=['tutorial_pembelian_IPO'])
async def support_dan_resistance(message):
    await message.reply(json_modul['modul']['5']['5.3']+"\n\nMelanjutkan ke Ch 5.4. /dividen")

@dp.message_handler(commands=['dividen'])
async def trend(message):
    await message.reply(json_modul['modul']['5']['5.4']+"\n\nMelanjutkan ke Ch 5.5. /stock_split")

@dp.message_handler(commands=['stock_split'])
async def chart_pattern(message):
    await message.reply(json_modul['modul']['5']['5.5']+"\n\nMelanjutkan ke Ch 5.6. /right_issue")

@dp.message_handler(commands=['right_issue'])
async def taktik_trading(message):
    await message.reply(json_modul['modul']['5']['5.6'] + "\n\nMelanjutkan ke Ch 5.7. /tender_offer")

@dp.message_handler(commands=['tender_offer'])
async def money_management(message):
    await message.reply(json_modul['modul']['5']['5.7']+"\n\nKembali ke list modul /moduls")

# @dp.message_handler(commands=['harga'])
# async def harga(message):

@dp.message_handler(commands=['saham_per_sektor'])
async def saham_per_sektor(message):
    reply = 'Daftar sektor yang tercatat mentransaksikan saham di BEI:\n'
    for sektor in list(saham_per_sector.keys()):
        reply = reply + "/" + sektor + "\n"
    await message.reply(reply)

@dp.message_handler(commands=["Barang_Konsumen_Primer"])
async def Barang_Konsumen_Primer(message):
    reply = "Saham yang terdapat dalam sektor Barang_Konsumen_Primer diantaranya :\n"
    for saham in saham_per_sector["Barang_Konsumen_Primer"]:
        reply = reply + saham + "\n"
    await message.reply(reply)

@dp.message_handler(commands=["Barang_Konsumen_Non_Primer_1"])
async def Barang_Konsumen_Non_Primer_1(message):
    reply = "Saham yang terdapat dalam sektor Barang_Konsumen_Non_Primer diantaranya :\n"
    for saham in saham_per_sector["Barang_Konsumen_Non_Primer_1"]:
        reply = reply + saham + "\n"
    await message.reply(reply)

@dp.message_handler(commands=["Barang_Konsumen_Non_Primer_2"])
async def Barang_Konsumen_Non_Primer_2(message):
    reply = "Saham yang terdapat dalam sektor Barang_Konsumen_Non_Primer diantaranya :\n"
    for saham in saham_per_sector["Barang_Konsumen_Non_Primer_2"]:
        reply = reply + saham + "\n"
    await message.reply(reply)

@dp.message_handler(commands=["Finansial"])
async def Finansial(message):
    reply = "Saham yang terdapat dalam sektor Finansial diantaranya :\n"
    for saham in saham_per_sector["Finansial"]:
        reply = reply + saham + "\n"
    await message.reply(reply)

@dp.message_handler(commands=["Industrial"])
async def Industrial(message):
    reply = "Saham yang terdapat dalam sektor Industrial diantaranya :\n"
    for saham in saham_per_sector["Industrial"]:
        reply = reply + saham + "\n"
    await message.reply(reply)

@dp.message_handler(commands=["Infrastruktur"])
async def Infrastruktur(message):
    reply = "Saham yang terdapat dalam sektor Infrastruktur diantaranya :\n"
    for saham in saham_per_sector["Infrastruktur"]:
        reply = reply + saham + "\n"
    await message.reply(reply)

@dp.message_handler(commands=["Properti_Dan_Real_Estate"])
async def Properti_Dan_Real_Estate(message):
    reply = "Saham yang terdapat dalam sektor Properti_Dan_Real_Estate diantaranya :\n"
    for saham in saham_per_sector["Properti_Dan_Real_Estate"]:
        reply = reply + saham + "\n"
    await message.reply(reply)

@dp.message_handler(commands=["Energi"])
async def Energi(message):
    reply = "Saham yang terdapat dalam sektor Energi diantaranya :\n"
    for saham in saham_per_sector["Energi"]:
        reply = reply + saham + "\n"
    await message.reply(reply)

@dp.message_handler(commands=["Barang_Baku"])
async def Barang_Baku(message):
    reply = "Saham yang terdapat dalam sektor Barang_Baku diantaranya :\n"
    for saham in saham_per_sector["Barang_Baku"]:
        reply = reply + saham + "\n"
    await message.reply(reply)

@dp.message_handler(commands=["Transportasi_Dan_Logistik"])
async def Transportasi_Dan_Logistik(message):
    reply = "Saham yang terdapat dalam sektor Transportasi_Dan_Logistik diantaranya :\n"
    for saham in saham_per_sector["Transportasi_Dan_Logistik"]:
        reply = reply + saham + "\n"
    await message.reply(reply)

@dp.message_handler(commands=["Teknologi"])
async def Teknologi(message):
    reply = "Saham yang terdapat dalam sektor Teknologi diantaranya :\n"
    for saham in saham_per_sector["Teknologi"]:
        reply = reply + saham + "\n"
    await message.reply(reply)

@dp.message_handler(commands=['Kesehatan'])
async def Kesehatan(message):
    reply = "Saham yang terdapat dalam sektor Kesehatan diantaranya :\n"
    for saham in saham_per_sector['Kesehatan']:
        reply = reply + saham + "\n"
    await message.reply(reply)

@dp.message_handler(commands=['logo'])
async def logo(message):
    await message.reply("ini logo")
    await message.answer_photo('https://avatars.githubusercontent.com/u/62240649?v=4')
    await bot.send_photo(chat_id=message.chat.id, photo=open('./Dataset/laboratory.jpg', 'rb'))

@dp.message_handler(commands=['harga'])
async def harga(message):
    try:
        kode_saham = str(message.text).split()[1].upper()
    except IndexError:
        await message.reply("Silahkan memasukkan kode saham yang ingin diketahui harganya hari ini dengan format :\n"
                            "/harga KODE_SAHAM (Contoh: /harga BBCA)\n"
                            "(Harga yang ditampilkan memiliki delay harga antara 3-10 menit)")
    else:
        get_harga_saham = requests.get("https://api.goapi.id/v1/stock/idx/prices?api_key="+key_goAPI+"&symbols="+kode_saham)
        harga_saham = get_harga_saham.json()['data']['results']
        if (len(harga_saham)==0):
            await message.reply("Maaf, saham dengan kode saham '" +kode_saham+"' tidak ditemukan")
        else:
            harga_saham = harga_saham[0]
            perubahan_harga_saham = int(harga_saham.get("close"))-int(harga_saham.get("open"))
            symbol = "+" if perubahan_harga_saham>0 else "-"
            perubahan_harga_saham = symbol + str(abs(perubahan_harga_saham))
            persentase_perubahan_harga_saham = str(format(abs(((int(harga_saham.get("close"))/int(harga_saham.get("open")))*100)-100),".2f"))
            persentase_perubahan_harga_saham = symbol+persentase_perubahan_harga_saham
            from datetime import datetime
            now = datetime.now()
            date_string = now.strftime("%d/%m/%Y")
            time_string = now.strftime("%H:%M:%S")
            # t = time.localtime()
            # current_time = time.strftime("%H:%M:%S", t)
            reply = harga_saham.get('ticker')+" ("+date_string+")\n"+time_string+" WIB\n"+"Rp "+harga_saham.get('close')+"\n"+perubahan_harga_saham+" ("+persentase_perubahan_harga_saham+"%)\n"
            await message.reply(reply)

@dp.message_handler(commands=['info'])
async def info(message):
    try:
        kode_saham = str(message.text).split()[1].upper()
    except IndexError:
        await message.reply("Silahkan memasukkan kode saham perusahaan yang ingin diketahui informasinya dengan format :\n"
                            "/info KODE_SAHAM (Contoh: /info BBCA)")
    else:
        if (kode_saham not in json_info_saham):
            await message.reply("Maaf, saya tidak memiliki informasi mengenai perusahaan dengan kode saham '" +kode_saham+"'")
        else:
            exception_keys = ['is_listed','is_delisted','is_suspend']
            info_saham_kode = json_info_saham[kode_saham]  # info saham spesifik dari semua info saham
            info_keys = list(info_saham_kode.keys())
            reply = ""
            for key in info_keys:
                if key not in exception_keys:
                    reply = reply + str(key).title() + ":\n" + str(info_saham_kode[key]) + "\n\n"
            await message.reply(reply)

@dp.message_handler()
async def tanya_saham(message):
    if str(message.text)=='/tanya':
        await message.reply("Silahkan langsung menanyakan pertanyaan seputar saham yang ingin ditanyakan (Contoh: Apa itu saham?)")
    else:
        qa_start = time.monotonic()
        response = qa.Handle_Response(message.text)
        qa_end = time.monotonic()
        exec_time = round(qa_end-qa_start,2)
        print("Execution Time :",exec_time,"Detik")
        await message.reply(response)

if __name__ == '__main__':
    global qa, json_info_saham, json_modul, key_goAPI, saham_per_sector
    qa = QA()
    oov = OOV()
    oov.Menambahkan_Pertanyaan_OOV()
    with open('info_saham.json', 'r') as open_info_saham:
        json_info_saham = json.load(open_info_saham) #info semua saham
    with open('modul.json', 'r') as open_modul:
        json_modul = json.load(open_modul)
    with open('saham_per_sector.json', 'r') as open_saham_per_sector:
        saham_per_sector = json.load(open_saham_per_sector)
    key_goAPI = "TbZXoTsXp9Va3NKwVub6JBRjQkFkGi"
    print("MelekSaham is activated")
    executor.start_polling(dp)