from transformers import BertTokenizer, BertModel
import torch

class TextEmbedding:
    def __init__(self):
        # 初始化模型和分词器
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def get_embedding(self, text):
        """
        获取给定文本的嵌入表示。
    
        参数：
            text (str): 需要获取嵌入表示的文本。
    
        返回：
            numpy.ndarray: 文本的嵌入表示，形状为 (1, 嵌入维度)。
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()
    

    def text_embedding(self, datas):
        """
        根据给定的文本数据列表，计算并返回每个文本的嵌入向量。
    
        参数：
            datas (list): 一个包含文本数据的列表。
    
        返回：
            embeddings (list): 一个包含每个文本嵌入向量的列表。
        """
        embeddings = [self.get_embedding(text) for text in datas]
        return embeddings
    
