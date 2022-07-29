from typing import List, Tuple

import nltk
import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch import nn
from torch.types import Device
from torchtext.vocab import build_vocab_from_iterator, Vocab

INIT_DATA = [["გამარჯობა", "როგორ", "ხარ?"], ["რავი", "კარგად", "შენ?"]]

CHECK_NON_EMPTY_FN = lambda s: len(s) > 0


class GeorgianWord2Vec:
	def __init__(self) -> None:
		print("Initializing data")
		nltk.download('punkt')
		self.__model_name = "word2vec.model"
		model = Word2Vec(sentences=INIT_DATA, vector_size=100, window=5, min_count=1,
						 workers=4, epochs=3)
		model.save(self.__model_name)
		print("Model created!")

	# Returns word2vec model, trained by initial dataset
	def get_model(self) -> Word2Vec:
		return Word2Vec.load(self.__model_name)

	# Gets the file path and trains the model using its sentences
	def train(self, file_path: str) -> None:
		model = self.get_model()
		sentences = self.__convert_file_into_input(file_path)
		print(sentences[0])
		model.build_vocab(sentences, update=True)
		model.train(sentences, total_examples=model.corpus_count,
					epochs=model.epochs)
		model.save(self.__model_name)

	# Returns vector corresponding to passed word
	def get_vector(self, word: str) -> List[int]:
		return Word2Vec.load(self.__model_name).wv[word]

	def get_embeddings(self):
		return list(self.get_model().wv.index_to_key)

	@staticmethod
	def __convert_file_into_input(file_path: str) -> List[List[str]]:
		with open(file_path, 'r') as f:
			data = f.read()

		sentences = sent_tokenize(data)
		print(sentences)
		return [[word.strip() for word in sentence.split(" ")] for sentence in
				sentences]


class GeorgianLanguageDatasetLoader:
	def __init__(self, file_path: str, batch_size=10, device="cpu") -> None:
		data = open(file_path, encoding="utf-8").read().split("\n")
		df = pd.DataFrame(data, columns=["sentence"])
		train, test = train_test_split(df["sentence"].tolist(), train_size=0.6,
									   test_size=0.2, random_state=0)
		train, valid = train_test_split(train, test_size=0.25, random_state=0)
		train_iter = train
		valid_iter = valid
		test_iter = test
		self.vocab = build_vocab_from_iterator(map(word_tokenize, train_iter),
											   specials=['<unk>'])
		self.vocab.set_default_index(self.vocab['<unk>'])
		self.train_data = self.__data_process(train_iter)
		self.valid_data = self.__data_process(valid_iter)
		self.test_data = self.__data_process(test_iter)
		self.text_pipeline = self.__create_text_pipeline()

		eval_batch_size = batch_size // 2
		self.train_data_batched = batchify(self.train_data,
										   batch_size,
										   device)  # shape [seq_len, batch_size]
		self.val_data_batched = batchify(self.valid_data, eval_batch_size, device)
		# test_data = batchify(test_data, eval_batch_size)

	@staticmethod
	def __yield_tokens(train):
		for line in train:
			sentences = sent_tokenize(line)
			print(word_tokenize(line))
			words = [[word.strip() for word in sentence.split(" ")] for sentence in
					 sentences]
			for word in words:
				yield word

	def __data_process(self, raw_text_iter):
		"""Converts raw text into a flat Tensor."""
		data = [torch.tensor(self.vocab(word_tokenize(item)), dtype=torch.long) for item
				in
				raw_text_iter]
		return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

	def __create_text_pipeline(self):
		return lambda x: self.vocab(word_tokenize(x))

	def get_vocabulary(self):
		return self.vocab

	def get_data(self):
		return self.train_data, self.valid_data, self.test_data

	def get_batched_data(self):
		return self.train_data_batched, self.val_data_batched

	def get_text_pipeline(self):
		return self.text_pipeline


def batchify(data: Tensor, bsz: int, device: Device) -> Tensor:
	"""Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [bsz, N // bsz]
    """
	seq_len = data.size(0) // bsz
	data = data[:seq_len * bsz]
	data = data.view(bsz, seq_len)
	return data.to(device)


def get_batch(source: Tensor, i: int, bptt=10) -> Tuple[Tensor, Tensor]:
	"""
    Args:
        source: Tensor, shape [batch_size, full_seq_len]
        i: int

    Returns:
        tuple (data, target), where data has shape [batch_size, seq_len] and
        target has shape [seq_len * batch_size]
    """
	seq_len = min(bptt, source.size(1) - i)
	data = source[:, i:i + seq_len]
	target = source[:, i + 1:i + 1 + seq_len]
	if data.shape != target.shape:
		return data, data
	return data, target


def create_emb_layer(weights_matrix, non_trainable=False):
	num_embeddings, embedding_dim = weights_matrix.size()
	emb_layer = nn.Embedding(num_embeddings, embedding_dim)
	emb_layer.load_state_dict({'weight': weights_matrix})
	if non_trainable:
		emb_layer.weight.requires_grad = False

	return emb_layer, num_embeddings, embedding_dim


class LSTMModel(nn.Module):
	def __init__(self, embedding_dim, hidden_size, vocab_size, device, num_layers=1,
				 embeddings=None):
		super().__init__()
		if embeddings is not None:
			self.emb, num_embeddings, embedding_dim = create_emb_layer(embeddings, True)
		else:
			self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim).to(device)

		self.input_size = embedding_dim
		self.hidden_size = hidden_size
		self.device = device
		self.num_layers = num_layers
		self.embedding_dropout = nn.Dropout(0.3)
		self.embeddings = embeddings

		self.lstm = nn.LSTM(input_size=embedding_dim,
							hidden_size=hidden_size,
							bias=True,
							batch_first=True,
							bidirectional=True,
							num_layers=num_layers).to(device)
		# use for bidirectional
		self.classifier = nn.Linear(hidden_size * 2, vocab_size).to(device)
		# self.classifier = nn.Linear(hidden_size, vocab_size).to(device)

	def forward(self, inp, prev_state):
		emb = self.emb(inp)
		emb = self.embedding_dropout(emb)

		out, state = self.lstm(emb, prev_state)
		logits = self.classifier(out)
		return logits, state

	def init_state(self, sequence_length):
		# use for bidirectional
		return (torch.zeros(self.num_layers * 2, sequence_length, self.hidden_size).to(self.device),
				torch.zeros(self.num_layers * 2, sequence_length, self.hidden_size).to(self.device))
		# return (
		# torch.zeros(self.num_layers, sequence_length, self.hidden_size).to(self.device),
		# torch.zeros(self.num_layers, sequence_length, self.hidden_size).to(self.device))


def compute_perplexity(model, batched_data, batch_size, bptt=20):
	model.eval()

	with torch.no_grad():  # tells Pytorch not to store values of intermediate computations for backward pass because we not gonna need gradients.
		loss = 0
		batch_count = 0
		state_h, state_c = model.init_state(batch_size)
		for i in range(0, batched_data.size(1) - 1, bptt):
			x, y = get_batch(batched_data, i)
			y_pred, (state_h, state_c) = model(x, (state_h, state_c))
			loss += torch.nn.functional.cross_entropy(
				y_pred.reshape(x.size(0) * x.size(1), -1), y.reshape(-1)).item()
			batch_count += 1

			state_h = state_h.detach()
			state_c = state_c.detach()

	model.train()

	return np.exp(loss / batch_count)


def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group['lr']


def train_loop(model, batch_data, batch_size, bptt=20):
	model.train()

	# if we have dropout in neural network, we need to use model.train()

	# L2 L=Loss+|W| , Weight decay L = Loss, DL=DLoss + dW
	# we add weight decay (L2 regularization) to avoid overfitting. weight decay = L2
	# try removing it and see what happens.
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

	# we will reduce initial learning rate by 'lr=lr*factor' every time validation perplexity doesn't improve within certain range.
	# details here https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
	lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5,
															  min_lr=1e-6, patience=10)

	crit = nn.CrossEntropyLoss(reduction='mean')

	# L = average(L[i]), L[i] = Cross Entropy(y_hat=Softmax(classifier(RNN_hiden_state)), one_hot(next word))

	# calc_accuracy(model)

	it = 1
	total_loss = 0
	curr_perplexity = None
	perplexity = None
	for epoch in range(500):
		state_h, state_c = model.init_state(batch_size)
		for i in range(0, batch_data.size(0) - 1, bptt):

			x, y = get_batch(batch_data, i)

			# if we don't do this, pytorch will accumulate gradients (summation) from previous backprop to next backprop and so on...
			# this behaviour is useful when we can't fit many samples in memory but we need to compute gradient on large batch. In this case, we simply accumulate gradient and do backprop
			# after enough samples has been seen.
			optimizer.zero_grad()  # best way to understand it is to google it up.
			# print(x.shape, y.shape)

			# do forward pass, will save intermediate computations of the graph for later backprop use.
			y_pred, (state_h, state_c) = model(x, (state_h, state_c))
			# y_pred has shape (batch_size, max_seq_len, vocab_size), y has shape (batch_size, max_seq_len), and we
			# need to compute average Cross Entropy across batch and sequence dimensions. For this, we first reshape tensors accordingly.
			loss = crit(y_pred.reshape(x.size(0) * x.size(1), -1), y.reshape(-1))
			total_loss += loss.item()  # (zero dimentional tensor).item() -> one number

			state_h = state_h.detach()
			state_c = state_c.detach()

			# running backprop.
			loss.backward()

			# doing gradient descent step.
			optimizer.step()

			# we are logging current loss/perplexity in every 100 iteration (in every 100 batch)
			if it % 100 == 0:

				# computing validation set perplexity in every 500 iteration.
				if it % 500 == 0:
					curr_perplexity = compute_perplexity(model, batch_data, batch_size,
														 bptt)

					lr_scheduler.step(curr_perplexity)

					# making checkpoint of best model weights.
					if not perplexity or curr_perplexity < perplexity:
						torch.save(model.state_dict(), 'model')
						perplexity = curr_perplexity

				print('Epoch', epoch + 1, '| Iter', it, '| Avg Train Loss',
					  total_loss / 100, '| Dev Perplexity', curr_perplexity, '| LR ',
					  get_lr(optimizer))
				total_loss = 0

			it += 1


def generate_text(model, device: Device, vocab: Vocab, context: str,
				  length: int = 10, ):
	model = model.to(device)
	model.eval()

	state_h, state_c = model.init_state(1)

	res = context.split(' ')
	for i in range(length):
		x = vocab(word_tokenize(''.join(res)))
		x = torch.LongTensor(x).view(1, len(x)).to(device)
		y_pred, (state_h, state_c) = model(x, (state_h, state_c))
		y_pred = torch.topk(y_pred[0, -1], dim=-1, k=5).indices

		# we filter out <unk> and <eos> tokens having indices 0 and 3 respectively
		y_pred = [i.item() for i in y_pred if i.item() not in [0, 3]]
		res.append(vocab.get_itos()[np.random.choice(y_pred)])

	return ' '.join(res)
