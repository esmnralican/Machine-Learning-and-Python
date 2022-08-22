# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 09:43:19 2022

@author: esmnralican
"""

import pandas as pd

#%% 
# import twitter data
data = pd.read_csv(r"gender-classifier-DFE-791531.csv" , encoding = "latin1" ) # encoding = "latin1" : csv dosyası içersinde latin alfebesi kullanıldı.
data = pd.concat([data.gender , data.description] , axis =1)

data.dropna(axis = 0 , inplace = True)
#dropna(axis = 0) : nan değerlere sahip olan column'ları direk siler 
#inplace = True : data = data.dropna(axis = 0) -> inplace tekra bir değişkene atama durumunu ortadan kaldrırıyor.

# classification algoritmaları string değerleri tercih etmez bu nedenle dataset içerindeki değerler int'a dönüştürülmeli.
data.gender = [ 1 if each ==  "female"  else 0 if each == "male" else 2 for each in data.gender] # female : 1 , male : 0


#%%
#cleaning data , regular expression(RE)
import re

first_description = data.description[4]
# "[^a-zA-Z]" , " "  : ^a-zA-Z : verilen aralığı bulma ve geri kalanı " " (boşluk) ile değiştir
description = re.sub("[^a-zA-Z]" , " ", first_description) # a-zA-Z  : verilen aralığı bul , ^a-zA-Z : verilen aralığı bulma

description = description.lower() # tüm karakterler küçük harfe çevrildi.


#%%
#stopwords(irrelavent words) gereksiz kelimeler
import nltk # natural language tool kit
nltk.download("stopwords") # irrelavent kelimeler corpus adlı bir dosyaya indiriliyor.
from nltk.corpus import stopwords # ardından corpus klasöründen import ediliyor.

# şimdi sırada cümlede bulunan kelimeleri ayırmak var 
description = description.split() # her bir kelime listeye aktarıldı.
# split metodu yalnızca boşlukları ayırır : shouln't do -> shouldn't | do 



# tokenizer metodu kelimeleri ayırır :  shouln't do -> should | n't | do
# split() alternatif : tokenizer
description = nltk.word_tokenize(description)

#%%
# irrelavent kelimeleri çıkar
description = [ word for word in description if not word in set(stopwords.words("english"))] # description içersindeki irrelevant word'ler çıkartıldı.

#%%
# lemmatazation : köklerine ayırma 
import nltk 
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()
description = [ lemma.lemmatize(word) for word in description]

description = " ".join(description) # temizlenen kelimeler tekrar cümle haline getirildi.

#%%
description_list = []

for description in data.description: 
    description = re.sub("[^a-zA-Z]" , " ", description) 
    description = description.lower()
    description = nltk.word_tokenize(description)
    description = [ word for word in description if not word in set(stopwords.words("english"))] # description içersindeki irrelevant word'ler çıkartıldı.
    lemma = WordNetLemmatizer()
    description = [ lemma.lemmatize(word) for word in description]
    description = " ".join(description) # temizlenen kelimeler tekrar cümle haline getirildi.
    description_list.append(description)
    
    
#%%
# bag of words
from sklearn.feature_extraction.text import CountVectorizer # bag of words yartamak için kullanılan method

max_features = 500 # .csv dosyasında ortalama 24000 adet kelime bulunduğu varsayılır ise max_features = 500(en çok kullanılan 500 adedi) ile 500 adedi üzerinden çalışmalara devam edeceğiz(performans açısından).
 
count_vectorizer = CountVectorizer(max_features = max_features , stop_words = "english" )

sparce_matrix =  count_vectorizer.fit_transform(description_list).toarray()

print(" en sık kullanılan {} kelimeler {}".format(max_features, count_vectorizer.get_feature_names())) # en sık kullanılan 500 adet kelime yazdırıldı.

#%%
y = data.iloc[:,0].values # male or female classes | .values : numpy'a dönüştütüldü
x = sparce_matrix
# train test split
from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.1 , random_state= 42)

#%%
# x_train ve y_train'i kullanarak  model eğitimi gerçekleştiriliyor. Ardından  x_test'i kullanarak test edilecek ardından çıkan sonuç y_test ile karşılaştırılacak.
 
#%%
#naive bayes

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()   

nb.fit(x_train , y_train )

#%%
#prediction
y_pred = nb.predict(x_test)
 
print( "accuracy", nb.score(y_pred.reshape(-1,1) ,y_test ) )

#%%
from sklearn.metrics import accuracy_score 
accuracy = accuracy_score(y_test , y_pred )
print( "accuracy {} " .format(accuracy))



