import nltk
import operator
import requests
from bs4 import BeautifulSoup
import re
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk import Text
from collections import Counter
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations

retokenize = RegexpTokenizer("[\w]+")
#영화 각각의 imdb 사이트 주소
r1 = requests.get('https://www.imdb.com/title/tt5095030/plotsummary?ref_=tt_stry_pl#synopsis')
r2 = requests.get('https://www.imdb.com/title/tt2015381/plotsummary?ref_=tt_stry_pl#synopsis')
r3 = requests.get('https://www.imdb.com/title/tt1825683/plotsummary?ref_=tt_stry_pl#synopsis')
r4 = requests.get('https://www.imdb.com/title/tt2250912/plotsummary?ref_=tt_stry_pl#synopsis')
r5 = requests.get('https://www.imdb.com/title/tt0478970/plotsummary?ref_=tt_stry_pl#synopsis')
r6 = requests.get('https://www.imdb.com/title/tt1211837/plotsummary?ref_=tt_stry_pl#synopsis')
r7 = requests.get('https://www.imdb.com/title/tt3896198/plotsummary?ref_=tt_stry_pl#synopsis')
r8 = requests.get('https://www.imdb.com/title/tt3501632/plotsummary?ref_=tt_stry_pl#synopsis')
r9 = requests.get('https://www.imdb.com/title/tt0800080/plotsummary?ref_=tt_stry_pl#synopsis')
r10 = requests.get('https://www.imdb.com/title/tt1981115/plotsummary?ref_=tt_stry_pl#synopsis')
r11 = requests.get('https://www.imdb.com/title/tt0800369/plotsummary?ref_=tt_stry_pl#synopsis')
r12 = requests.get('https://www.imdb.com/title/tt0848228/plotsummary?ref_=tt_stry_pl#synopsis')
r13 = requests.get('https://www.imdb.com/title/tt2385427/plotsummary?ref_=tt_stry_pl#synopsis')
r14 = requests.get('https://www.imdb.com/title/tt3498820/plotsummary?ref_=tt_stry_pl#synopsis')
r15 = requests.get('https://www.imdb.com/title/tt0458339/plotsummary?ref_=tt_stry_pl#synopsis')
r16 = requests.get('https://www.imdb.com/title/tt1843866/plotsummary?ref_=tt_stry_pl#synopsis')
r17 = requests.get('https://www.imdb.com/title/tt1300854/plotsummary?ref_=tt_stry_pl#synopsis')
r18 = requests.get('https://www.imdb.com/title/tt1228705/plotsummary?ref_=tt_stry_pl#synopsis')
r19 = requests.get('https://www.imdb.com/title/tt0371746/plotsummary?ref_=tt_stry_pl#synopsis')
r20 = requests.get('https://www.imdb.com/title/tt4154756/plotsummary?ref_=tt_stry_pl#synopsis')
r21 = requests.get('https://www.imdb.com/title/tt4154796/plotsummary?ref_=tt_stry_pl#synopsis')
r22 = requests.get('https://www.imdb.com/title/tt4154664/plotsummary?ref_=tt_stry_pl#synopsis')
r23 = requests.get('https://www.imdb.com/title/tt6320628/plotsummary?ref_=tt_stry_pl#synopsis')
c1 = r1.content
c2 = r2.content
c3 = r3.content
c4 = r4.content
c5 = r5.content
c6 = r6.content
c7 = r7.content
c8 = r8.content
c9 = r9.content
c10 = r10.content
c11 = r11.content
c12 = r12.content
c13 = r13.content
c14 = r14.content
c15 = r15.content
c16 = r16.content
c17 = r17.content
c18 = r18.content
c19 = r19.content
c20 = r20.content
c21 = r21.content
c22 = r22.content
c23 = r23.content
html1 = BeautifulSoup(c1, 'html.parser')
html2 = BeautifulSoup(c2, 'html.parser')
html3 = BeautifulSoup(c3, 'html.parser')
html4 = BeautifulSoup(c4, 'html.parser')
html5 = BeautifulSoup(c5, 'html.parser')
html6 = BeautifulSoup(c6, 'html.parser')
html7 = BeautifulSoup(c7, 'html.parser')
html8 = BeautifulSoup(c8, 'html.parser')
html9 = BeautifulSoup(c9, 'html.parser')
html10 = BeautifulSoup(c10, 'html.parser')
html11 = BeautifulSoup(c11, 'html.parser')
html12 = BeautifulSoup(c12, 'html.parser')
html13 = BeautifulSoup(c13, 'html.parser')
html14 = BeautifulSoup(c14, 'html.parser')
html15 = BeautifulSoup(c15, 'html.parser')
html16 = BeautifulSoup(c16, 'html.parser')
html17 = BeautifulSoup(c17, 'html.parser')
html18 = BeautifulSoup(c18, 'html.parser')
html19 = BeautifulSoup(c19, 'html.parser')
html20 = BeautifulSoup(c20, 'html.parser')
html21 = BeautifulSoup(c21, 'html.parser')
html22 = BeautifulSoup(c22, 'html.parser')
html23 = BeautifulSoup(c23, 'html.parser')

#영화 각각의 시놉시스 불러오기

location1 = html1.select('#synopsis-py3994279')
location2 = html2.select('#synopsis-py3256133 ')
location3 = html3.select('#synopsis-py3748588 ')
location4 = html4.select('#synopsis-py3259532 ')
location5 = html5.select('#synopsis-py3217570 ')
location6 = html6.select('#synopsis-py3277428 ')
location7 = html7.select('#synopsis-py3270263 ')
location8 = html8.select('#synopsis-py3573283')
location9 = html9.select('#synopsis-py3229019 ')
location10 = html10.select('#synopsis-py3255367 ')
location11 = html11.select('#synopsis-py3229118 ')
location12 = html12.select('#synopsis-py3230367 ')
location13 = html13.select('#synopsis-py3261753 ')
location14 = html14.select('#synopsis-py3268444 ')
location15 = html15.select('#synopsis-py3216861 ')
location16 = html16.select('#synopsis-py3253321')
location17 = html17.select('#synopsis-py3241770 ')
location18 = html18.select('#synopsis-py3239566 ')
location19 = html19.select('#synopsis-py3276086 ')
location20 = html20.select('#synopsis-py3881753 ')
location21 = html21.select('#synopsis-py4495824 ')
location22 = html22.select('#synopsis-py4408383 ')
location23 = html23.select('#synopsis-py4615763')
print(location1)
exit()
all_location = [location1,location2,location3,location4,location5,location6,location7,location8,location9,location10,location11,location12,location13,location14,location15,location16,location17,location18,location19,location20,location21,location22,location23]
all_location
all_location2 = []
#tokenizing을 통해 명사를 추출
for x in all_location :
    x = str(list(x))
    x = re.sub('[>|.|(|/|"|=|<|,|)|_|[|-|"]', '', x)
    x = re.sub('[-]', ' ', x)
    x = retokenize.tokenize(x)
    all_location2.append(x)
for i in range(len(all_location)) :
    all_location[i] = all_location2[i]
eachwordset = []
wordsetfreq = []

#단어의 빈도수를 계산하기 위해 등장한 모든 단어들 모으기
for i in all_location :
    wordsetfreq+=i
len(wordsetfreq)

#단어의 등장횟수를 세기 위해 set으로 만들어주기
for i in all_location :
    i = set(i)
    eachwordset+=i
eachwordset = list(set(eachwordset))

#분석을 진행해봤을 때, 중요도를 따지는 것에 필요한 것으로 보이는 NNP를 추출
# NNP를 뽑아내서 분석에 이용하려고 했으나, 단어 set에서 바로 NNP를 뽑아내게 되면, NNP말고 다른 품사도 같이 나오는 현상이 발생하여, 먼저 다른 품사를 불용어로 처리한 후, NNP를 추출
NPwords = []
tag_list = pos_tag(eachwordset)
CC_list = [t[0] for t in tag_list if t[1] == 'CC']
CD_list = [t[0] for t in tag_list if t[1] == 'CD']
RB_list = [t[0] for t in tag_list if t[1] == 'RB']
DT_list = [t[0] for t in tag_list if t[1] == 'DT']
IN_list = [t[0] for t in tag_list if t[1] == 'IN']
JJ_list = [t[0] for t in tag_list if t[1] == 'JJ']
JJS_list = [t[0] for t in tag_list if t[1] == 'JJS']
JJR_list = [t[0] for t in tag_list if t[1] == 'JJR']
PRP_list = [t[0] for t in tag_list if t[1] == 'PRP']
TO_list = [t[0] for t in tag_list if t[1] == 'TO']
VB_list = [t[0] for t in tag_list if t[1] == 'VB']
VBP_list = [t[0] for t in tag_list if t[1] == 'VBP']
VBD_list = [t[0] for t in tag_list if t[1] == 'VBD']
VBN_list = [t[0] for t in tag_list if t[1] == 'VBN']
VBG_list = [t[0] for t in tag_list if t[1] == 'VBG']
VBZ_list = [t[0] for t in tag_list if t[1] == 'VBZ']
RP_list = [t[0] for t in tag_list if t[1] == 'RP']
RB_list = [t[0] for t in tag_list if t[1] == 'RB']
RBR_list = [t[0] for t in tag_list if t[1] == 'RBR']
MD_list = [t[0] for t in tag_list if t[1] == 'MD']
IN_list = [t[0] for t in tag_list if t[1] == 'IN']
WP_list = [t[0] for t in tag_list if t[1] == 'WP']
WRB_list = [t[0] for t in tag_list if t[1] == 'WRB']
WDT_list = [t[0] for t in tag_list if t[1] == 'WDT']
EX_list = [t[0] for t in tag_list if t[1] == 'EX']
FW_list = [t[0] for t in tag_list if t[1] == 'FW']
adding_stopwords = CC_list+RB_list+IN_list+JJ_list+PRP_list+TO_list+VB_list+VBP_list+DT_list+FW_list+WDT_list+EX_list+WRB_list+IN_list+RP_list+CD_list+MD_list+RBR_list+RB_list+WP_list+VBZ_list+VBG_list+VBN_list+VBD_list+DT_list+VBP_list+VB_list+TO_list+PRP_list+JJR_list+JJS_list+JJ_list+IN_list
adding_stopwords = list(set(adding_stopwords))
stop_words = stopwords.words('english')
for i in range(len(adding_stopwords)) :
    stop_words.append(adding_stopwords[i])
stop_words = list(set(stop_words))
for x in eachwordset :
    if x not in stop_words:
        NPwords.append(x)
# 위 과정으로 NNP를 뽑아냈는데도, 걸러지지 않는 품사가 있어서, 다시 pos_tag를 사용하여 NNP만 추출
tag_NPwords = pos_tag(NPwords)
NPwords = [t[0] for t in tag_NPwords if t[1] == 'NNP']

# 위의 과정에서 추출한 명사들이 등장한 빈도를 세는 매트릭스를 생성
freq0 = [0 for x in range(len(NPwords))]
freq1 = [0 for x in range(len(NPwords))]
freq2 = [0 for x in range(len(NPwords))]
freq3 = [0 for x in range(len(NPwords))]
freq4 = [0 for x in range(len(NPwords))]
freq5 = [0 for x in range(len(NPwords))]
freq6 = [0 for x in range(len(NPwords))]
freq7 = [0 for x in range(len(NPwords))]
freq8 = [0 for x in range(len(NPwords))]
freq9 = [0 for x in range(len(NPwords))]
freq10 = [0 for x in range(len(NPwords))]
freq11 = [0 for x in range(len(NPwords))]
freq12 = [0 for x in range(len(NPwords))]
freq13 = [0 for x in range(len(NPwords))]
freq14 = [0 for x in range(len(NPwords))]
freq15 = [0 for x in range(len(NPwords))]
freq16 = [0 for x in range(len(NPwords))]
freq17 = [0 for x in range(len(NPwords))]
freq18 = [0 for x in range(len(NPwords))]
freq19 = [0 for x in range(len(NPwords))]
freq20 = [0 for x in range(len(NPwords))]
freq21 = [0 for x in range(len(NPwords))]
freq22 = [0 for x in range(len(NPwords))]

#각각의 작품에서 분석에 쓸 단어들이 등장한 빈도수를 계산
for j in range(0,23) :
    for x in all_location[j]:
        try :
            ind = NPwords.index(x)
            eval('freq'+str(j))[ind]+=1
        except :
            continue

freq0 = pd.Series(freq0)
freq1 = pd.Series(freq1)
freq2 = pd.Series(freq2)
freq3 = pd.Series(freq3)
freq4 = pd.Series(freq4)
freq5 = pd.Series(freq5)
freq6 = pd.Series(freq6)
freq7 = pd.Series(freq7)
freq8 = pd.Series(freq8)
freq9 = pd.Series(freq9)
freq10= pd.Series(freq10)
freq11 = pd.Series(freq11)
freq12= pd.Series(freq12)
freq13 = pd.Series(freq13)
freq14 = pd.Series(freq14)
freq15 = pd.Series(freq15)
freq16 = pd.Series(freq16)
freq17 = pd.Series(freq17)
freq18 = pd.Series(freq18)
freq19 = pd.Series(freq19)
freq20 = pd.Series(freq20)
freq21 = pd.Series(freq21)
freq22 = pd.Series(freq22)
# 분석에 쓰이는 형태의 매트릭스를 만드는 과정
prevmat = pd.concat([freq0,freq1,freq2,freq3,freq4,freq5,freq6,freq7,freq8,freq9,freq10,freq11,freq12,freq13,freq14,freq15,freq16,freq17,freq18,freq19,freq20,freq21,freq22],axis=1)
result = np.dot(prevmat,prevmat.T)
weightedM = pd.DataFrame(result)
for i in range(len(NPwords)) :
    weightedM.rename(columns={i:NPwords[i]},inplace=True)
    weightedM.rename(index={i:NPwords[i]},inplace=True)

#####################################################################################################################
#####################################################################################################################
# 네트워크 생성
weightedG = nx.Graph()
unweightedG = nx.Graph()

# 가중치 있는 그래프
for x in NPwords:
    weightedG.add_node(x)
for x in list(combinations(NPwords, 2)):
    x1, y1 = x
    if weightedM.loc[x] == 0 :
        continue
    elif weightedM.loc[x] != 0:
        weightedG.add_edge(x1, y1,weight=weightedM.loc[x])

# 가중치 없는 그래프
for x in NPwords:
    unweightedG.add_node(x)
for x in list(combinations(NPwords, 2)):
    x1, y1 = x
    if weightedM.loc[x] == 0 :
        continue
    elif weightedM.loc[x] != 0:
        unweightedG.add_edge(x1, y1)
# 1. Degree
# Weighted
weightedDegree = dict(nx.degree(weightedG))
weightedDegree_keys = list(weightedDegree.keys())
weightedDegree_values = list(weightedDegree.values())
df_weightedDegree = pd.DataFrame({'NODE_ID': weightedDegree_keys, 'DC': weightedDegree_values})
df_weightedDegree.sort_values(by=['DC'],ascending=False,inplace=True)
df_weightedDegree.plot(kind='bar')
# UnWeighted
unweightedDegree = dict(nx.degree(unweightedG))
unweightedDegree_keys = list(unweightedDegree.keys())
unweightedDegree_values = list(unweightedDegree.values())
df_unweightedDegree = pd.DataFrame({'NODE_ID': unweightedDegree_keys, 'DC': unweightedDegree_values})
df_unweightedDegree.sort_values(by=['DC'],ascending=False,inplace=True)
df_unweightedDegree.plot(kind='bar')

# 가중치를 준 그래프와 가중치를 주지 않은 네트워크를 비교해봤을 때, 차이가 발생하지 않으므로, 명사의 중요도 파악은 weighted graph를 사용
# 중요도 파악을 위해 상위 20개 명사 추출
df_weightedDegree.iloc[:20,:]

# 2. Betweeness Centrality
# Weighted
weightedBetweeness = dict(nx.betweenness_centrality(weightedG))
weightedBetweeness_keys = list(weightedBetweeness.keys())
weightedBetweeness_values = list(weightedBetweeness.values())
df_weightedBetweeness = pd.DataFrame({'NODE_ID': weightedBetweeness_keys, 'BC': weightedBetweeness_values})
df_weightedBetweeness.sort_values(by=['BC'],ascending=False,inplace=True)
df_weightedBetweeness.plot(kind='bar')
# UnWeighted
unweightedBetweeness = dict(nx.betweenness_centrality(unweightedG))
unweightedBetweeness_keys = list(unweightedBetweeness.keys())
unweightedBetweeness_values = list(unweightedBetweeness.values())
df_unweightedBetweeness = pd.DataFrame({'NODE_ID': unweightedBetweeness_keys, 'BC': unweightedBetweeness_values})
df_unweightedBetweeness.sort_values(by=['BC'],ascending=False,inplace=True)
df_unweightedBetweeness.plot(kind='bar')

# 중요도 파악을 위해 상위 20개 명사 추출
df_weightedBetweeness.iloc[:20,:]

# 3. Closeness Centrality
# Weighted
weightedClose = dict(nx.closeness_centrality(weightedG))
weightedClose_keys = list(weightedClose.keys())
weightedClose_values = list(weightedClose.values())
df_weightedClose = pd.DataFrame({'NODE_ID': weightedClose_keys, 'CC': weightedClose_values})
df_weightedClose.sort_values(by=['CC'],ascending=False,inplace=True)
df_weightedClose.plot(kind='bar')
# UnWeighted
unweightedClose = dict(nx.closeness_centrality(unweightedG))
unweightedClose_keys = list(unweightedClose.keys())
unweightedClose_values = list(unweightedClose.values())
df_unweightedClose = pd.DataFrame({'NODE_ID': unweightedClose_keys, 'CC': unweightedClose_values})
df_unweightedClose.sort_values(by=['CC'],ascending=False,inplace=True)
df_unweightedClose.plot(kind='bar')

# 중요도 파악을 위해 상위 20개 명사 추출
df_weightedClose.iloc[:20,:]

# 4. 각각의 단어 빈도수를 계산
for i in range(len(NPwords)) :
    prevmat.rename(index={i:NPwords[i]},inplace=True)
df_frequency = prevmat.sum(axis=1)
df_frequency.sort_values(inplace=True,ascending=False)
# 중요도 파악을 위해 상위 30개 명사 추출
df_frequency[:30]

plt.plot(df_frequency)
