# coding: utf-8
from scipy import stats
from sklearn import datasets
from sklearn import svm
import numpy as np
import re
#text
#input: [[句子1, label1]], [句子2, label], ...]
file = open('data/taggedsents.txt','r')
taggedtext = file.readlines()
taggedtext = [i.rstrip() for i in taggedtext]
taggedtext = [i.split('\t') for i in taggedtext]
text = [i[0] for i in taggedtext]
labels = [i[1] for i in taggedtext]
file.close()
#print (len(labels), labels[:2],text[:2], taggedtext[:2])
#print(text)
# combine all features as a list
f1 = (r'你看')
f2 = (r'不覺得她|那')
f3 = (r'(她|他|那個)(.*?)(喔|噢|哦)')
f4 = (r'胖|粗魯|沒氣質|沒有氣質|太矮|脾氣|脾氣|差很多|很沒有|不適合|任性|差|爛|粗|好胖|很大|整形')
f5 = (r'吵|煩|管|嘮叨|黏')
f6 = (r'公主(病)?|獨立')
f7 = (r'亂花錢|金錢觀|價值觀|錢')
f8 = (r'嗎$')
f9 = (r'^這(.*?)是不是')
f10 = (r'會不會')
f11 = (r'怎麼辦')
#f12 = (r'(.*)$怎麼辦(.*)$')
f13 = (r'(.*?)還是(.*?)')
f14 = (r'(.*?)比較(.*?)')
f15 = (r'^你覺得')
f16 = (r'真|最|超')
f17 = (r'真的(.*?)喔')
f18 = (r'(對|可以|好)(.*)(阿|啊)')
f19 = (r'不會|才不會')
f20 = (r'(很*|沒*)(.*)(壓|呀)')
f21 = (r'真的|完全')
f22 = (r'可以')
f23 = (r'(嗎|啊|阿|了)$')
f24 = (r'嗯+|喔+')
f25 = (r'都')
f26 = (r'覺得(.*?)(她|他|我)(.*?)(他|她|我)')
f27 = (r'(.*?)比不上')
f28 = (r'(.*?)差(.*?)')
f29 = (r'(.*?)不好(.*?)')
f30 = (r'(.*?)不覺得(.*?)')
f31 = (r'誰')
f32 = (r'什麼|為何|為什麼')
f33 = (r'認識|知道')
f34 = (r'一起')
f35 = (r'陪')
f36 = (r'只(有)?找你')
f37 = (r'還是(.*?)去')
f38 = (r'(他|她|你)(.*?)去')

#f39 = (word_length <=3)

feature_list = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,f31,f32,f33,f34,f35,f36,f37,f38]
#print (feature_list)
#### append elements in nested list 

#user_input = input("快陶啊! 她說:")
#def baby(user_input):
#    input_list = []
#    input_list.append(user_input)
#    return input_list
#print (input_list)


#########################################
def sent_score(user_sent):
    new_text = text+[user_sent]
    matched = []
    for i in feature_list:
        f = []
        for sent in new_text:
            pattern = re.compile(i)
            match = pattern.findall(sent)
            f.append(len(match))
        matched.append(f)
    #len(matched)
    x = np.array(matched)
    newmatched = x.T
    #print (newmatched)
    scored_match = stats.zscore(newmatched)
    #matchlist = scored_match.tolist()
    # == taco #
    matchlist = scored_match.tolist()
    matchlist_1 = matchlist[0:-1]
    matchlist_2 = matchlist[-1]
    featuredict = list(zip(labels, matchlist_1))
    data = [a[1] for a in featuredict]
    target = [a[0] for a in featuredict]
    clf = svm.SVC()

    X,y = data, target  
    clf.fit(X,y)

    # end taco #
    return clf.predict(matchlist_2).tolist()[0]
