#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from stat import *
import nltk
from nltk.corpus import stopwords
import numpy as np
import sklearn.feature_extraction.text as text
import sklearn.metrics.pairwise as pairwise
from sklearn import decomposition
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 

def AbreDocumento(arquivo):
	texto = arquivo.split('. ') #Texto separado pro frase 
	texto = '\n'.join(texto)
	
	arquivo2 = open('C:/Users/Patrícia/Dropbox/POC/A/texto.txt', 'w') #Caminho para salvar arquivo
	arquivo2.writelines(texto)
	arquivo2.close()
	
	arquivo = open('C:/Users/Patrícia/Dropbox/POC/A/texto.txt', encoding = "utf8", errors = 'ignore')
	
	#print(arquivo)
	
	return (arquivo)

def FechaDocumento(arquivo):
	arquivo.close()

def FechaDocumento(arquivo):
	arquivo.close()
	
def Tokenizacao(arquivo):
	#Tokenização
	vector = text.CountVectorizer(input = 'arquivo', stop_words = 'english', min_df = 1, strip_accents = 'unicode')
	arqArray = vector.fit_transform(arquivo).toarray() #Vetor de frequencia de tokens
	
	vocabulario = np.array(vector.get_feature_names())
	
	numWords = arqArray.shape[1]
	
	FechaDocumento(arquivo)
	#print (arqArray)
	
	return(arqArray, numWords)
	
def Query(arquivo, nome, numWords):
	#Separa strings do nome da feature
	nome = nome.lower()
	nome = nome.split()
	tamNome = len(nome) 
	
	#Contagem de tokens pelo nome da feature
	arquivoAux = arquivo.read()
	arquivoAux = arquivoAux.lower()
	arqToken = arquivoAux.split()
	#print (arqToken)

	tokenName = []
	cont = 0
	
	for x in range(0, tamNome):
		for i in arqToken:
			if i == nome[x]:
				cont = cont + 1;
		tokenName.append(cont)	
		cont = 0

	for i in range(tamNome, numWords):
		tokenName.append('0') 
		
	FechaDocumento(arquivo)
	#print(tokenName)
	
	return(tokenName, numWords)

def Similaridade(arqArray, tokenName):
	arqArray[np.isnan(arqArray)] = 0;
	simCosseno = pairwise.cosine_similarity(tokenName, arqArray)
	
	return(simCosseno)
	
def Similaridade2(arqArray, tokenName, numWords):
	tokenName = (tokenName[0])
	tamTokenName = len(tokenName) 
	newTokenName = []
	
	for i in range(0, tamTokenName):
		newTokenName.append(tokenName[i])
		
	for i in range(tamTokenName, numWords):
		newTokenName.append('0') 
	
	arqArray[np.isnan(arqArray)] = 0;
	simCosseno = pairwise.cosine_similarity(newTokenName, arqArray)
	
	return(simCosseno)
	
def NMF(arquivo, nt):
	vector = text.CountVectorizer(input='arquivo', stop_words='english', min_df=1, strip_accents='unicode')
	arqArray = vector.fit_transform(arquivo).toarray()
	vocabulario = np.array(vector.get_feature_names())
	
	ntw = arqArray.shape[0]
	
	#NMF
	num_topics = nt
	num_top_words = ntw
	#Decomposição
	clf = decomposition.NMF(n_components=num_topics, random_state=1)
	doctopic = clf.fit_transform(arqArray)

	topic_words = []
	for topic in clf.components_:
		word_idx = np.argsort(topic)[::-1][0:num_top_words]
		topic_words.append([vocabulario[i] for i in word_idx])
		
	with np.errstate(invalid='ignore'):
		doctopic = doctopic / np.sum(doctopic, axis=1, keepdims=True)
	
	
	#print(doctopic)

	FechaDocumento(arquivo)

	return(doctopic)

def Resultado(saidaSimilaridade, arquivo, top, nomeDoc):
	listaSentencas = arquivo.readlines()
	x = (len(listaSentencas))
	aux = 0
	rank = []
	sentVet = []
	
	for i in range(0, x):
		for j in range(0, x):
			if (saidaSimilaridade[0][i] > saidaSimilaridade[0][j]):
				aux = i
		rank.append(aux)
		
	rank = list(set(rank))

	
	
	for i in range(0, x):
		listaSentencas[i] = listaSentencas[i].replace("\n", ".")

	print("Sumário:")
	if (len(rank)) >= 2:
		for i in range(0, top):
			print(listaSentencas[rank[i]])
			
	if (len(rank)) == 1:
		print(listaSentencas[rank[0]])
		result = (random.choice(listaSentencas))
		print(result)
		
	if (len(rank)) == 0:
		for i in range(0, top):
			result = (random.choice(listaSentencas))
			if result not in sent:
				sent.append(result)
				print(result)
		
		
	return (sentVet)
		
if __name__ == "__main__":
	nomeDoc = (sys.argv[1])
	numSent = int(sys.argv[2])
	textoDoc = (sys.argv[3])
	texto = AbreDocumento(textoDoc)
	saidaTokenizacao1, saidaTokenizacao2 = Tokenizacao(texto)
	texto = AbreDocumento(textoDoc)
	saidaQuery, numWords = Query(texto, nomeDoc, saidaTokenizacao2)
	saidaSimilaridade = Similaridade(saidaTokenizacao1, saidaQuery)
	texto = AbreDocumento(textoDoc)
	saidaNMF = NMF(texto, saidaTokenizacao2)
	saidaSimilaridade = Similaridade2(saidaNMF, saidaSimilaridade, numWords)
	texto = AbreDocumento(textoDoc)
	Resultado(saidaSimilaridade, texto, numSent, nomeDoc)