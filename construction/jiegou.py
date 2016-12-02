# encoding:utf-8

import pymongo
import collections
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import chardet

client = pymongo.MongoClient('localhost', 27017)
RecruitDBD = client['RecruitDBDC']
recurit_url = RecruitDBD['recuritdbd']
job_describedb_python = RecruitDBD['job_describedb_python']

job_describedb_python.remove()

f = open('yaoqiu.txt','r')
f2 = open('yaoqiu.txt','r')
yaoqiuguilei = {}
yaoqiu = {'岗位要求':1,'岗位职责':2,'工作地址':3,'应聘条件':4,'薪资水平':5,'工作经验':6,'公司简介':7,'优先考虑':8}
yaoqiu2 = {j:i for i,j in yaoqiu.items()}
for i in f:
    a = i.strip().split()
    # print chardet.detect(a[0])
    key = a[0].decode('gb2312').encode('utf-8')
    # print key
    yaoqiuguilei[key] = yaoqiu2[int(a[1])]

job_describe = {}
for i in f2:
    job_describe[i.strip().split()[0]] = ''
# print(request_list)
for i in recurit_url.find():
    # print('/////////////////////////////////////////////////////////////////////////////////////')
    # print(i['job_description'].strip())
    a = i['job_name'].lower()
    if 'python' in a:
        des_count = {}
        des_count2 = collections.defaultdict(lambda :[])
        job_split = {}
        job_split2 = {}
        job_description = ''.join(i['job_description']).strip()
        job_description = str(job_description).replace('查看职位地图','')
        job_description = job_description.encode('utf-8')
        len_des = len(job_description)

        for j in job_describe:
            # print chardet.detect(j)
            # print chardet.detect(job_description)
            j = j.decode('gb2312').encode('utf-8')
            if j in job_description:
                des_count[j] = job_description.index(j)
        des_count = sorted(des_count.items(),key=lambda a:a[1])
        # print(des_count)
        for i in range(len(des_count)):
            try:
                des_count2[des_count[i][0]].append(des_count[i+1][1])
                des_count2[des_count[i][0]].append(des_count[i][1])
            except:
                des_count2[des_count[i][0]].append(len_des)
                des_count2[des_count[i][0]].append(des_count[i][1])
        des_count2 = sorted(des_count2.items(),key=lambda a:a[1][0])
        # print(des_count2)

        for k in des_count2:
            job_split[k[0]] = job_description[k[1][1]+len(k[0]):k[1][0]].strip()
        # if '任职资格：' in job_split:
        #     print(job_split['任职资格：'])
        for i,j in job_split.items():
            # print chardet.detect(i)
            # print chardet.detect(yaoqiuguilei[i])
            job_split2[yaoqiuguilei[i]] = j

        # print(job_split2['岗位职责'])
        if '岗位要求'in job_split2 or '应聘条件' in job_split2 or '岗位职责' in job_split2 or '优先考虑' in job_split2:

            job_describedb_python.insert_one(job_split2)