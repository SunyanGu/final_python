# encoding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import time
from gensim import corpora,models,similarities
import collections
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import cv2.cv as cv
from Bio.Cluster import kmedoids
import matplotlib.pyplot as plt
import chardet

class Document_distance(object):
    def __init__(self):
        # self.filename = 'f_cleaned.txt'

        self.filename = '../create_require/f_cleaned.txt'

    def w2v(self):
        texts = self.create_raw_documents()
        model = models.word2vec.Word2Vec(texts, min_count=1,window=2)  # 训练skip-gram模型; 默认window=5
        # f_2 = open('result/comment_acc.vector', 'wb')
        f_2 = open('comment_acc.vector', 'wb')
        model.save_word2vec_format(f_2, binary=False)
        return model

    def load_w2v_model(self):

        # model = models.word2vec.Word2Vec.load_word2vec_format("result/comment_acc.vector")
        model = models.word2vec.Word2Vec.load_word2vec_format("comment_acc.vector")
        nvs = zip(model.index2word,model.syn0)
        w2v_dict = dict((word,vec) for word,vec in nvs)
        return w2v_dict

    def Euclidean_distance(self,arr1,arr2):
        return np.sqrt(np.sum(np.square(arr1 - arr2)))

    def create_w2v_distence_matrix(self):
        pass

    def create_Bow(self):
        texts = self.create_raw_documents()
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        dictionary = dictionary.token2id

        # index_word = self.create_index_word()
        # for i in corpus:
        #     for j in i:
        #         print index_word[j[0]]
        return corpus,dictionary

    def create_word_index(self):
        corpus, word_index = self.create_Bow()
        return word_index

    def create_index_word(self):
        word_index = self.create_word_index()
        index_word = {j:i for i,j in word_index.items()}
        return index_word


    #注意：corpus含有空项;corpus是随机产生的
    def create_nBow(self):
        corpus, dictionary = self.create_Bow()
        Col = len(dictionary)
        Row = len(corpus)
        nBow = np.zeros((Row,Col))
        row = 0
        for i in corpus:
            for j in i:
                nBow[row][j[0]] = j[1]
            sum_Rol = np.sum(nBow[row,:])
            nBow[row,:] = nBow[row,:] / float(sum_Rol)
            row += 1
        return nBow

    def create_raw_documents(self):
        f = open(self.filename,'r')
        texts = [[word for word in sentence.strip().split()]for sentence in f]
        word_list = []
        for i in texts:
            for j in i:
                word_list.append(j)
        # print(len(set(word_list)))
        return texts

    def create_sentence_index(self):
        index_sentence = {}
        f = open(self.filename,'r')
        for i,j in enumerate(f.readlines()):
            index_sentence[i] = j.strip()
        sentence_index = [(j,i) for i,j in index_sentence.items()]
        return index_sentence,sentence_index

    def amount_sentence(self):
        index_sentence, sentence_index = self.create_sentence_index()
        return len(sentence_index)

    def sentence2word(self,sentence):
        pass

    def calc_EMD_distance(self,sentence1ID,sentence2ID,index_sentence,word_index,nBow,w2v_dict):
        sen1 = index_sentence[sentence1ID]
        sen2 = index_sentence[sentence2ID]
        sen1_list = []
        # print sen1
        for i in sen1.split():
            # print i.encode('utf-8')
            num = word_index[unicode(i)]
            # print word_index[unicode(i)]
            # print nBow[0][num]
            # print w2v_dict[unicode(i)]
            w2v = np.array(w2v_dict[unicode(i)],dtype=np.float)
            # print np.array([nBow[0][num]],dtype=np.float)
            W = np.array([nBow[sentence1ID][num]],dtype=np.float)
            vec = np.concatenate((W,w2v))
            # print vec
            sen1_list.append(vec)
        sen1_array = np.array(sen1_list,dtype=np.float)

        sen2_list = []
        # print sen2

        for i in sen2.split():
            # print i.encode('utf-8')
            num = word_index[unicode(i)]
            # print word_index[unicode(i)]
            # print nBow[1][num]
            # print w2v_dict[unicode(i)]
            w2v = np.array(w2v_dict[unicode(i)],dtype=np.float)
            # print np.array([nBow[1][num]],dtype=np.float)
            W = np.array([nBow[sentence2ID][num]],dtype=np.float)
            vec = np.concatenate((W, w2v))
            # print vec
            sen2_list.append(vec)
        sen2_array = np.array(sen2_list,dtype=np.float)

        sen1_64 = cv.fromarray(sen1_array)
        sen1_32 = cv.CreateMat(sen1_64.rows, sen1_64.cols, cv.CV_32FC1)
        sen2_64 = cv.fromarray(sen2_array)
        sen2_32 = cv.CreateMat(sen2_64.rows, sen2_64.cols, cv.CV_32FC1)

        cv.Convert(sen1_64, sen1_32)
        cv.Convert(sen2_64, sen2_32)

        # print (cv.CalcEMD2(sen1_32, sen2_32, cv.CV_DIST_L2, lower_bound=float('inf')))
        EMD_distance = cv.CalcEMD2(sen1_32, sen2_32, cv.CV_DIST_L2, lower_bound=float('inf'))
        # print time.time() - start
        return EMD_distance

    def create_distance_matrix(self):
        # sentenceID = self.create_sentence_index()
        amount = self.amount_sentence()
        index_sentence, sentence_index = self.create_sentence_index()
        word_index = self.create_word_index()
        nBow = self.create_nBow()
        w2v_dict = self.load_w2v_model()
        # print sentenceID
        distance_matrix = np.zeros((amount,amount))
        count = 0
        for i in range(amount):
            for j in range(amount):
                # start = time.time()
                distance_matrix[i][j] = self.calc_EMD_distance(i,j,index_sentence,word_index,nBow,w2v_dict)                  #need 2 seconds
                count += 1
                if count % 100000 == 0:
                    print count
                # print time.time() - start
        np.save("distance_matrix.npy",distance_matrix)
        # print distance_matrix
        return distance_matrix

    def k_medoids(self):
        distance = np.load("distance_matrix.npy")
        clusterid, error, nfound = kmedoids(distance,nclusters=10)
        cluster_dict = collections.defaultdict(lambda: [])
        num = 0
        for i in clusterid:
            cluster_dict[i].append(num)
            num += 1
        return cluster_dict

    def print_result(self):
        index_sentence, sentence_index = self.create_sentence_index()
        cluster_dict = self.k_medoids()
        num = 0
        f = open('result_10.txt','w')
        for i,j in cluster_dict.items():
            f.write('------------------------------------------')
            f.write(str(num)+'\n')
            for m in j:
                f.write(index_sentence[m]+'\n')
            num += 1

    def get_elbow(self):
        cluster_point = [i for i in range(5,400,5)]
        print cluster_point
        distance = np.load("distance_matrix.npy")
        error_list = []
        for num in cluster_point:
            clusterid, error, nfound = kmedoids(distance, nclusters=num)
            error_list.append(error)
        plt.plot(cluster_point, error_list, 'r')
        plt.show()

    def get_cluster_contain(self):
        index_sentence, sentence_index = self.create_sentence_index()
        cluster_dict = self.k_medoids()
        cluster_contain = collections.defaultdict(lambda :[])
        for i, j in cluster_dict.items():
            for m in j:
                cluster_contain[i].append(index_sentence[m])

        return cluster_contain

    def get_most_important(self):
        index_sentence, sentence_index = self.create_sentence_index()
        cluster_dict = self.k_medoids()
        distance = np.load("distance_matrix.npy")
        most_important_dict = collections.defaultdict(lambda :[])
        least_important_dict = collections.defaultdict(lambda :[])
        for i,j in cluster_dict.items():
            nearest = distance[i].argsort()[:5][0::]
            farthest = distance[i].argsort()[-5:][::-1]
            for m in nearest:
                print distance[i][m]
                most_important_dict[i].append(index_sentence[m])
            for m in farthest:
                print distance[i][m]
                least_important_dict[i].append(index_sentence[m])
        return most_important_dict,least_important_dict

    def print_important_sentence(self):
        index_sentence, sentence_index = self.create_sentence_index()
        most_important_dict, least_important_dict = self.get_most_important()
        f_important = open('most_important.txt','w')
        f_unimportant = open('unimportant.txt','w')
        for i,j in most_important_dict.items():
            f_important.write('/////////////////////////////////////////////////\n')
            f_important.write('中心句是：\n'+ str(index_sentence[i])+'\n\n')
            for m in j:
                f_important.write(m+'\n')

        for i, j in least_important_dict.items():
            f_unimportant.write('/////////////////////////////////////////////////\n')
            f_unimportant.write('中心句是：\n' + str(index_sentence[i]) + '\n\n')
            for m in j:
                f_unimportant.write(m + '\n')

    def get_main_content(self):
        index_sentence, sentence_index = self.create_sentence_index()
        cluster_dict = self.k_medoids()
        distance = np.load("distance_matrix.npy")
        most_important_dict = collections.defaultdict(lambda: [])
        for i, j in cluster_dict.items():
            length = int(len(j) * 0.9)
            nearest = distance[i].argsort()[:length][0::]
            for m in nearest:
                # print distance[i][m]
                most_important_dict[i].append(index_sentence[m])
        return most_important_dict



    def lda(self):
        index_sentence, sentence_index = self.create_sentence_index()
        # cluster_dict = self.k_medoids()
        most_important_dict = self.get_main_content()

        # document = []
        # for i,j in most_important_dict.items():
        #     text = []
        #     for sent in j:
        #         sentence =  sent.strip().split()
        #         text.extend(sentence)
        #     document.append(text)



        document = []
        for i,j in most_important_dict.items():
            text_sentence = []
            for sent in j:
                sentence =  sent.strip().split()
                text_sentence.append(sentence)
        #     document.append(text)
        # doc = []
        # doc.append(document[0])


        dic = corpora.Dictionary(text_sentence)  # 构造词典
        corpus = [dic.doc2bow(text) for text in text_sentence]  # 每个text 对应的稀疏向量
        tfidf = models.TfidfModel(corpus)  # 统计tfidf
        print "lda"
        corpus_tfidf = tfidf[corpus]  # 得到每个文本的tfidf向量，稀疏矩阵
        lda = models.LdaModel(corpus_tfidf, id2word=dic, num_topics=1)
        for i in range(1):
            print lda.print_topic(i)
        pass
        # corpus_lda = lda[corpus_tfidf]  # 每个文本对应的LDA向量，稀疏的，元素值是隶属与对应序数类的权重
        # print "lda"
        #
        # for doc in corpus_lda:
        #     print doc


    def run(self):
            self.w2v()
            self.create_distance_matrix()
            self.print_result()

document_distance = Document_distance()
# document_distance.w2v()
# document_distance.load_w2v_model()
# document_distance.create_Bow()
# document_distance.create_sentence_index()
# document_distance.create_nBow()
# document_distance.calc_EMD_distance(1,2)
# document_distance.create_distance_matrix()
# document_distance.amount_sentence()

# document_distance.print_result()


# document_distance.run()


# document_distance.get_elbow()


# document_distance.get_most_important()
# document_distance.print_important_sentence()

document_distance.lda()

# document_distance.get_main_content()