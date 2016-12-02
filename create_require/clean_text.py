# -*- coding:UTF-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import jieba
import re

f_cleaned_text = open('f_cleaned.txt','w')
f_stopword = open('my_stopword.txt','r')
stopword_list = []
for i in f_stopword:
    stopword_list.append(i.strip())


f_require = open('require.txt','r')


# 长句
# for i in f_require:
#     # print(i)
#     seg_list = jieba.cut(i)  # 默认是精确模式
#     acc_str = " ".join(seg_list)
#     # print(acc_str.split())
#     # for j in stopword_list:
#     #     if j in acc_str.split():
#     #         acc_str.replace(j,'')
#     # print(acc_str)
#     texts = [word.lower() for word in acc_str.split() if word not in stopword_list]
#
#     # print(texts)
#     sentence_str = ''
#     for j in texts:
#         sentence_str += j+' '
#     print(sentence_str)
#     if len(sentence_str)>2:
#         f_cleaned_text.write(sentence_str+'\n')

# 短句
for i in f_require:

    # print i
    sentence = re.compile('、|，|；|,|。')
    sentence_split =  sentence.split(i.strip())
    short_sentence_list = []
    for short_sen in sentence_split:
        # print short_sen
        seg_list = jieba.cut(short_sen)  # 默认是精确模式
        acc_str = " ".join(seg_list)
        # print(acc_str.split())
        # for j in stopword_list:
        #     if j in acc_str.split():
        #         acc_str.replace(j,'')
        # print(acc_str)
        texts = [word.lower() for word in acc_str.split() if word not in stopword_list]

        # print(texts)
        sentence_str = ''
        for j in texts:
            sentence_str += j+' '
        # print(sentence_str)
        if len(sentence_str)>2:
            short_sentence_list.append(sentence_str)
    for i in set(short_sentence_list):
        f_cleaned_text.write(i+'\n')





