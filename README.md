

```python
# gcc -fPIC -shared lda_gibbs.c -o lda_gibbs.so
```


```python
from lda import LDA

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets.twenty_newsgroups import fetch_20newsgroups
```


```python
# download 20newsgroups dataset
dataset = fetch_20newsgroups(remove=('headers','footers','quotes'))
```


```python
# extract term-doc matrix of top 5000 words with 3 or more characters
pattern = '(?u)\\b[a-zA-Z]{3,}\\b'
cv = CountVectorizer(stop_words='english', max_features=5000, token_pattern=pattern)
doc_term = cv.fit_transform(dataset.data)
vocab = cv.get_feature_names()
```


```python
# 15 topic LDA model
n_topics = 15
lda = LDA(n_topics)
```


```python
# fit the model
lda.fit(doc_term)
```




    LDA with 15 topics




```python
# extract the two distributions we learned
user_topic = lda.theta
topic_word = lda.phi
```


```python
# print the top 10 words per topic
for topic in range(n_topics):
    idxs = topic_word[topic].argsort()[::-1][:10]
    print("Topic:",topic)
    print(', '.join([vocab[i] for i in idxs]))
```

    Topic: 0
    research, science, university, use, information, number, years, center, study, health
    Topic: 1
    know, thanks, like, problem, does, help, need, post, work, time
    Topic: 2
    god, jesus, believe, does, life, bible, christian, church, true, people
    Topic: 3
    people, israel, jews, armenian, world, war, turkish, history, government, armenians
    Topic: 4
    file, use, program, window, set, output, using, display, application, entry
    Topic: 5
    key, use, chip, used, encryption, security, public, keys, bit, government
    Topic: 6
    think, don, people, point, question, say, way, yes, make, actually
    Topic: 7
    said, didn, people, know, did, time, went, came, told, day
    Topic: 8
    car, used, new, power, price, bike, old, ground, sale, good
    Topic: 9
    gun, people, president, state, right, going, government, law, control, states
    Topic: 10
    drive, windows, card, dos, disk, bit, scsi, hard, memory, mac
    Topic: 11
    just, like, don, good, think, really, better, lot, make, thing
    Topic: 12
    max, space, bhj, giz, nasa, air, new, earth, launch, chz
    Topic: 13
    game, year, team, play, games, season, win, period, hockey, league
    Topic: 14
    edu, com, available, mail, list, ftp, software, version, send, information
