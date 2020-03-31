from tkinter import *
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot

print('package imported...')

df=pd.read_csv('dataset/Restaurant_Reviews.txt',delimiter='\t')
punc=string.punctuation
stop_words=stopwords.words('english')
stop_words.remove('not')
ps=PorterStemmer()
cv=CountVectorizer(binary=True)
pca=PCA(.99)
log=LogisticRegression()

print('objects created...')

def mypredict():
	msg=e.get()
	msg2=clean_text(msg)
	test_x=cv.transform([msg2]).toarray()
	test_x=pca.transform(test_x)
	pred=log.predict(test_x)
#	l3.configure(text=pred[0])
	if(pred[0]==0):
		l3.configure(text="Did not Like")
	if(pred[0]==1):
		l3.configure(text="Liked")

def clean_text(msg):
    msg=msg.lower()
    msg=re.sub(f'[{punc}]','',msg)
    words=word_tokenize(msg)
    new_words=[]
    for w in words:
        if(w not in stop_words):
            new_words.append(w)
    
    after_stem_words=[]
    for w in new_words:
        after_stem_words.append(ps.stem(w))
    clean_msg=' '.join(after_stem_words)
    return clean_msg


df['Review']=df.Review.apply(clean_text)
print('data cleaned...')
# df.Liked.value_counts().plot(kind='bar')


X=cv.fit_transform(df.Review).toarray()
new_X=pca.fit_transform(X)
y=df.iloc[:,-1].values
print('going for training...')
log.fit(new_X,y)
print('model trained....')

# def graph():
#     a=df.Liked.value_counts().plot(kind='bar')
#     l4.configure(a)

root=Tk()
root.state('zoomed')
root.configure(background='gray85')
root.title("Restaurant Reviews Project")

l1=Label(root,text='Restaurent Reviews',bg='gray85',fg='red', font='times 40 bold underline', anchor= 'center')
l1.place(x=450,y=20)


l2=Label(root,text='Enter your Review:',bg='gray85',fg='blue',font=('',20,'bold'))
l2.place(x=350,y=130)


e=Entry(root,font=('',20,''), justify = 'center')
e.place(x=650,y=130)


b=Button(root,text='Predict',font=('',20,''),relief= 'groove', command=mypredict)
b.place(x=600,y=220)


l3=Label(root,text='',bg='gray85',font='times 40 italic')
l3.place(x=550,y=320)

photo = PhotoImage(file='4.png')

p=Label(image=photo)
p.place(x=350,y= 420)


root.mainloop()