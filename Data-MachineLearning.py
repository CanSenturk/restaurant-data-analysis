#!/usr/bin/env python
# coding: utf-8

# In[249]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[250]:


tips=sns.load_dataset("tips")
df=tips.copy()


# In[251]:


df.head()#ilk 5satırı gösterir.


# In[252]:


df.tail()#son 5 satırı gösterir


# In[253]:


df.info()#veri seti ile ilgili bilgi getirir.


# In[254]:


df.describe()#veri setinin açıklamalarını görüyoruz


# In[255]:


df.describe().T#veri setinin transpozu alındı üsteki verilerle aynı şeyi yaptık 


# In[256]:


df.describe()#mean=ortalama, std=standard sapma


# In[257]:


#sütun adlarını değiştirme


# In[258]:


df.rename(columns={"total_bill":"fiyat",
                  "tip":"bahşiş",
                  "sex":"cinsiyet",
                  "smoker":"sigara",
                  "day":"gün",
                  "time":"zaman",
                  "size":"kişiler"}, inplace=True)#sütun adlarını değiştirme işlemi için, rename(....) komutu kullanılır. inplace=True komutu yaptığın değişkiliği kaydeder.


# In[259]:


df


# In[260]:


df.head()


# In[261]:


#içeriklerin değişmesi


# In[262]:


#cinsiyet


# In[263]:


df["cinsiyet"]=df.cinsiyet.map({"Female":"Kadın","Male":"Erkek"})#cinsiyet sütunundaki değerleri türkçeye çevirdik yani içeriği çevirdik. df["cinsiyet"] içine kayıt edersek içerikler kayıt olmuş olur.


# In[264]:


df


# In[265]:


df["sigara"]=df.sigara.map({"No":"Hayır","Yes":"Evet"})


# In[266]:


df


# In[267]:


df.cinsiyet.unique()#bu komut mesela cinsiyet kategorisinde kadın,erkek dışında tanımlama var mı onu söyler.unique=eşsiz anlamına gelir.


# In[268]:


df.sigara.unique()


# In[269]:


df.gün.unique()


# In[270]:


#sun=pazar
#sat=cumartesi
#thur=perşembe
#fri=cuma


# In[271]:


df["gün"]=df.gün.map({"Sun":"Pazar","Sat":"Cumartesi","Thur":"Perşembe","Fri":"Cuma"})#gün sütununu içeriğini değiştirip türkçeye çevirdik


# In[272]:


df


# In[273]:


df.zaman.unique()


# In[274]:


df["zaman"]=df.zaman.map({"Dinner":"Akşam Yemeği","Lunch":"Öğle Yemeği"})#zaman sütunundaki verileri türkçeye çevirerek sütun içindeki değerleri değiştirdik


# In[275]:


df


# In[276]:


df.head()


# In[277]:


df.gün.value_counts()#gün sütünü iiçinde kaç tane cumartesi,pazar,perş,cuma değeri olduğunu gösteren komuttur.


# In[278]:


df.sigara.value_counts()#sigara sütunu içinde kaçte evet hayır var onu gösteriyor


# In[279]:


df.zaman.value_counts()#zaman sütününda kaç tane akşam ve öğle yemeği yendiği gösterilmiş


# In[280]:


df.cinsiyet.value_counts()#cinsiyet sütununda kaçtane erkek,kadın vardır onu gösterdik


# In[281]:


df.gün.value_counts().plot()#bu komut grafikleştirmeye yarar.


# In[282]:


df.gün.value_counts().plot(kind="pie", autopct="%.1f%%")#kind="" komutu ile grafiğin türünü belirtebiliriz.pasta, çizgi, çubuk grafikleri vb., autopct="%.1f%% bu komut yüzdelik olarak göster demek"
plt.ylabel(" ")#y koordinatındaki yazıyı kaldırdık. label=etiket demek


# In[283]:


df.cinsiyet.value_counts().plot(kind="pie",autopct="%.1f%%")#cinsiyeti grafik şeklinde gösterdim.
plt.xlabel("CİNSİYET")# x koordinatına cinsiyet yazdık
plt.ylabel(" ");# y koordinatını boş bıraktım satır sonuna ; noktalı virgül koyarsan grafiğin üstündeki text,axes yazılarını siler.


# In[284]:


sns.barplot(x="gün",y="fiyat",data=df);#seabornun barplot çubuk grafiğini kullandık. x koordinatına gün, y koordinatına fiyat yazdırdık df verisini kullandık.


# In[285]:


sns.barplot(x="gün",y="kişiler",data=df);# gün ve kişileri karşılaştırıyoruz


# In[286]:


sns.barplot(x="gün",y="kişiler",hue="sigara",data=df);# hue komutu veriye 2. boyutu katar burada burada hangi günler daha kaç kişi sigara içmişi anlarız


# In[287]:


sns.barplot(x="zaman",y="fiyat",hue="kişiler",data=df);


# In[288]:


sns.barplot(x="zaman",y="fiyat",hue="cinsiyet",data=df);


# In[289]:


#cinsiyet sütunu içinde bulunan str ifadeleri sayısal ifadelere çevirmek


# In[290]:


df["cinsiyet"]=df.cinsiyet.map({"Kadın":1,"Erkek":2})


# In[291]:


df


# In[292]:


#sigara sütunu içinde bulunan str ifadeleri sayısal ifadelere çevirmek


# In[293]:


df.sigara.unique()


# In[294]:


df["sigara"]=df.sigara.map({"Hayır":2,"Evet":1})


# In[295]:


df


# In[296]:


df.gün.unique()


# In[297]:


df["gün"]=df.gün.map({"Perşembe":1,"Cuma":2,"Cumartesi":3,"Pazar":4})


# In[298]:


df


# In[299]:


#günler
#perşembe=1,Cuma=2,Cumartesi=3,Pazar=4


# In[300]:


#zaman


# In[301]:


df.zaman.unique()


# In[302]:


df["zaman"]=df.zaman.map({"Öğlen Yemeği":1,"Akşam Yemeği":2,})


# In[303]:


df


# In[304]:


df


# In[305]:


df


# In[306]:


#k en yakın komşu algoritması sınıflandırma ve regresyon analizlerinde çok kullanılır gözetimli öğrenmede algoritmasıdır.


# In[307]:


import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.metrics import mean_squared_error,r2_score

from sklearn import model_selection 
from sklearn.neighbors import KNeighborsRegressor

from warnings import filterwarnings
filterwarnings("ignore")


# In[308]:


#Train-Test(Eğitim-Test)


# In[309]:


df.head()


# In[310]:


#bağımlı bağımsız değişkenleri ayırmamız gerekli x ve y olarak ayırmalıyız


# In[311]:


#bağımsız değişken genelde x ile gösterilir


# In[312]:


X=df.drop(["fiyat"],axis=1)#burda bağımsız değişken olan fiyatı çıkarttık.drop o sütunu çıkarır axis=1 sütün, axis=0 satır
y=df["fiyat"]


# In[313]:


X


# In[314]:


y


# In[315]:


X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.20, random_state=42)#test_size test setinin yüzde kaçlık kesimini almamız gerektiğini belirtir. random_state ise veri setinin her noktasından veri alarak yüzde 20yi tamamla demek


# In[316]:


X_train[0:10]


# In[317]:


X_train.shape#X_train veri setinin 195 satır 6 sütundan oluştuğunu söylüyor


# In[318]:


X_test.shape#test veri setimizin 49 satır 6 sütundan oluştuğunu söylüyor


# In[319]:


y_train.shape#y eğitim setininde 195 satırdan oluştuğunu söylüyor


# In[320]:


y_test.shape#y test setimizinde 49 satırdan oluştuğunu söylüyor


# In[321]:


#model oluşturma


# In[322]:


knn_model=KNeighborsRegressor()# k en yakın komşu algoritması oluşturduk knn_model değerine atadık


# In[323]:


knn_model


# In[326]:


model = knn_model.fit(X_train, y_train)


# In[ ]:




