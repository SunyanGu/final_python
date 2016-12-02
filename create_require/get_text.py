# -*- coding:UTF-8 -*-

import pymongo
import re
import chardet
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

client = pymongo.MongoClient('localhost', 27017)
RecruitDBD = client['RecruitDBDC']
job_describedb_python = RecruitDBD['job_describedb_python']

f = open('require.txt','w')
for i in job_describedb_python.find({},{'_id':0,'应聘条件':1}):

    # if '应聘条件' in i.keys():
    #     if len(i['应聘条件']) != 0:
    #         require_list = []
    #         print(i['应聘条件'])
    #         str_yingpin = i['应聘条件']
    #         sentence = re.compile('\d+、|\d+\.|\d+）')
    #         print(sentence.split(str_yingpin)[1:])
    #         for j in sentence.split(str_yingpin)[1:]:
    #             try:
    #                 f.write(j.strip()+'\n')
    #             except:
    #                 pass
    if len(i.values()) != 0:
        require_list = []
        # print(i.values())
        str_yingpin = i.values()[0]
        # print str_yingpin
        sentence = re.compile(u'\d+\.|\d+、|\d+）|\d+，')
        # print  re.compile('、').split(str_yingpin)
        # a = re.findall(u'\d+\.|\d+、|\d+）|\d+，',str_yingpin,re.S)
        # print a
        # print(sentence.split(str_yingpin))
        for j in sentence.split(str_yingpin):
            # j = unicode(j)
            # print sentence.split(j)
            # print str(j)
            # print chardet.detect(j)
            if len(j) > 2:
                f.write(str(j).strip() + '\n')
                # try:
                #
                #     f.write(j.strip()+'\n')
                # except:
                #     print j.strip()
                #
                #     pass






