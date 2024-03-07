import pandas as pd
import numpy as np
import textwrap

import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("punkt")
nltk.download("stopwords")


def get_sentence_score(tfidf_row):
    # returns a mean of all the non-zero values of the tf-idf vector representation of the sentence
    x = tfidf_row[tfidf_row != 0]
    return x.mean()


def summarizer(text):
    sents = nltk.sent_tokenize(text.split("\n", 1)[1])

    if text:

        # print(sents)

        featurizer = TfidfVectorizer(
            stop_words=stopwords.words("english"),  # applying the stopwords removal
            norm="l1",
        )

        x = featurizer.fit_transform(sents)  # finally getting the tfidf array

        scores = np.zeros(len(sents))
        for i in range(0, len(sents)):
            score = get_sentence_score(x[i, :])
            scores[i] = score

        sort_index = np.argsort(-scores)

        summary = text.split("\n", 1)[0]

        for i in sort_index[:6]:
            summary += f"{scores[i]}{sents[i]}\n"

        try:
            print(summary)
        except Exception as e:
            print(e)
    else:
        print("Provide a text")


summarizer(
    "Could Yukos be a blessing in disguise?\n\nOther things being equal, the notion of entrepreneurs languishing in jail while their companies are sold off for a song ought to be bad for business.\n\nBut in the looking-glass world of modern Russia, the opposite might just be true, a new report* has argued. The study, from the Centre for Economic Policy Research, does not praise the rough handling of oil company Yukos. But it argues that more rigorous tax policing has benefited all Russian firms, even targets of the tax police. An increase in tax enforcement can increase the amount [of dividends and other income] outside shareholders will receive, even accounting for increased levels of taxation, the authors say.\n\nThe paper's reasoning is complex, and is based on a sophisticated model of the relationship between tax regimes and corporate governance - in particular, the propensity of management to steal from the company. The calculations demonstrated what many Russian analysts already knew: that increasing the tax rate increases the amount that managers steal, since undeclared income becomes relatively more valuable. In the West, meanwhile, higher tax rates translate far more smoothly into higher government revenues. On the other hand, increasing the rigour with which taxes are collected encourages companies to become more transparent, forcing them to be able to demonstrate their financial position far more accurately. The net result, the authors say, is that the extra amount companies pay in tax is more than compensated for by greater efficiency and financial soundness.\n\nAfter Vladimir Putin became president in 2000, he did not raise taxes, but put a lot of effort - too much, critics argue - into enforcement.\n\nSince then, the Russian stock market has more than trebled in value, a rise the authors attribute at least in part to the newly tough approach. The report highlights the case of Sibneft, a Russian oil company that came close to merging with Yukos last year. After Mr Putin came to power, the company's overall effective tax rate rose from 2.6% to 10.4%, and Sibneft was the target of a series of aggressive raids by fiscal police. But shareholders benefited hugely: Sibneft started to pay dividends - $53m in 2000 and almost $1bn in 2001 - and closed down the network of opaque subsidiaries it had previously used for siphoning off unofficial funds. According to the authors, although a variety of changes were sweeping through Russian industry at the time, the increase in tax enforcement is the only likely explanation for the change of fortunes at Sibneft and many of its peers.\n\nDoes this analysis make sense? In part, certainly. For all its faults, corporate Russia has become far more orderly and law-abiding since 2000. Companies have rushed to list their shares on international stock exchanges - something unthinkable in the wilder days of the 1990s - and most large firms now produce their accounts to international standards. Foreign direct investment, long negligible, is starting to flow in serious amounts - $7bn in 2003 - and stock market returns have been among the healthiest in Europe. But the authors' model does not quite cover all the complexities. For a start, the model assumes that the various parties have clearly-defined motivation: companies want to maximise profit, governments want to maximise tax revenue. In fact, the alarmingly close connections between big business and government in Russia - connections often greased by bribery - blur the apparently antagonistic relationship. Companies can, for example, persuade officials to overlook non-payment of taxes.\n\nAnd the authors' definition of tax enforcement seems unrealistically Western. Genuine, disinterested tax collection might well work wonders in Russia; the problem with recent examples has been the erratic and unpredictable way laws are enforced. The case against Yukos, for example, has moved in fits and starts, with little clarity from the government about its intentions, and little faith from investors that the letter of the law would be followed. As far as most commentators are concerned, the state is pursuing Yukos out of a political vendetta, rather than simply to enforce fiscal rectitude. Since Yukos' founder, Mikhail Khodorkovsky, was arrested a year ago, the Russian market has dropped by 10% - an indication that few investors feel optimistic about the salutary effect on corporate performance."
)
