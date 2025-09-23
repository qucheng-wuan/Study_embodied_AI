import math
from tqdm import tqdm
import torch
import torch.nn as nn
from numpy import random
import numpy as np


if torch.backends.mps.is_available():
    device = torch.device("mps")


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model) 
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)
        # self.pe = nn.Parameter(pe) 可以访问 而且可以参加梯度计算
        # self.pe = pe. #       仅仅可以访问
        #  model = PositionalEncoding(d_model=128, dropout=0.1)
# print(model.pe)  # 可以访问
# print(model.pe.requires_grad)  # False，不参与梯度计算

# 保存模型时，pe会被保存
# torch.save(model.state_dict(), 'model.pth')

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128 (seq_len,batch_size,d_model)
        """
        # 将x和positional encoding相加。
        x = x + self.pe[0,:x.size(0),:].unsqueeze(1).to(x.device).requires_grad_(False) #取前seq_len个位置编码  本来是（1，seq_len,d_model)转换(seq_len,1,d)
        return self.dropout(x)


class CopyTaskModel(nn.Module):
    def __init__(self, d_model=128):
        super(CopyTaskModel, self).__init__()

        # 定义词向量，词典数为10。我们不预测两位数。 embedding层的输入格式必须为(seq_len,batch_size)
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=128)
        # 定义Transformer。超参是我拍脑袋想的
        self.transformer = nn.Transformer(d_model=128, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=512)

        # 定义位置编码器
        self.positional_encoding = PositionalEncoding(d_model, dropout=0)

        # 定义最后的线性层，这里并没有用Softmax，因为没必要。
        # 因为后面的CrossEntropyLoss中自带了
        self.predictor = nn.Linear(d_model, 10)

    def forward(self, src, tgt):
        # 生成mask   这里输入的格式为(seq_len,batch_size)，因此token的长度为tgt的第0个维度，用.size()和.shape是一样的，返回的都是<class 'torch.Size'>
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[0]).to(src.device) # 这是一个静态函数，形参self随便填，填None也行，不影响结果

        src_key_padding_mask = CopyTaskModel.get_key_padding_mask(src).to(src.device)  # 因为实现的是静态类，所以不用self调用
        tgt_key_padding_mask = CopyTaskModel.get_key_padding_mask(tgt).to(src.device)

        # 对src和tgt进行编码
        src = self.embedding(src)
        tgt = self.embedding(tgt)

        # 给src和tgt的token增加位置信息
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        # 将准备好的数据送给transformer
        out = self.transformer(src, tgt,
                               tgt_mask=tgt_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask)
        """
        这里直接返回transformer的结果。因为训练和推理时的行为不一样，
        所以在该模型外再进行线性层的预测。
        """
        return out

    @staticmethod
    def get_key_padding_mask(tokens):
        """
        用于key_padding_mask
        """
        # key_padding_mask = torch.zeros(tokens.size(), dtype=torch.float)
        # key_padding_mask[tokens == 2] = (-math.inf)  # inf表示一个无穷大的数
        # # 由于transformer的输入参数中的src_key_padding_mask的格式为(batch_size,seq_len)，因此需要转置一下
        # key_padding_mask = key_padding_mask.transpose(0, 1)
        return (tokens == 2).transpose(0, 1).to(tokens.device).to(torch.bool) # (batch, seq_len)


# batch_size = 2 ; max_length = 16
def generate_random_batch(batch_size, max_length):
    src = []
    for i in range(batch_size):
        # 随机生成句子长度
        random_len = random.randint(1, max_length - 2)
        # 随机生成句子词汇，并在开头和结尾增加<bos>和<eos>  0表示起始符，1表示结束符，2表示填充符
        random_nums = [0] + [random.randint(3, 10) for _ in range(random_len)] + [1]
        # 如果句子长度不足max_length，进行填充（padding）
        random_nums = random_nums + [2] * (max_length - random_len - 2)
        src.append(random_nums)
    src = torch.LongTensor(src)  # src(batch_size,len_seq)(2,16)
    # src是源句子，tgt是目标句子(这里由于是序列预测而不是翻译，所以src=tgt)，tgt_y是label用于计算loss
    # tgt不要最后一个token
    tgt = src[:, :-1]  # (2,15)
    # tgt_y不要第一个的token
    tgt_y = src[:, 1:]  # (2.15)
    # 计算tgt_y，即要预测的有效token的数量
    n_tokens = (tgt_y != 2).sum()

    # tgt_y = tgt_y.transpose(0, 1)
    # 这里的n_tokens指的是我们要预测的tgt_y中有多少有效的token，后面计算loss要用
    return src, tgt, tgt_y, n_tokens


max_length = 16
model = CopyTaskModel().to(device)
criteria = nn.CrossEntropyLoss(ignore_index=2) # 忽略 <pad> 不是1 1是<eos>
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def train(EPOCH=1000):
    # 开始训练
    total_loss = 0
    for step in (range(EPOCH)):
        # 生成数据 torch.Tensor(batch_size,len_seq)
        # src(2,16) tgt(2,15) tgt_y(2,15)
        src, tgt, tgt_y, n_tokens = generate_random_batch(batch_size=32, max_length=max_length)

        # 清空梯度
        optimizer.zero_grad()
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        #tgt_y = tgt_y.transpose(0, 1)

        src = src.to(device); 
        tgt = tgt.to(device); 
        tgt_y = tgt_y.to(device)

        # 进行transformer的计算
        out = model(src, tgt)  # 传入的src和tgt应该为(seq_len,batch_size)格式
        # 将结果送给最后的线性层进行预测
        out = out.transpose(0, 1)
        out = model.predictor(out)  # (2,15,128)→(2,15,10)
        # y = torch.argmax(out[0], dim=1)


        """
        计算损失。由于训练时我们的是对所有的输出都进行预测，所以需要对out进行reshape一下。
                我们的out的Shape为(batch_size, 词数, 词典大小)，view之后变为：
                (batch_size*词数, 词典大小)。
                而在这些预测结果中，我们只需要对非<pad>部分进行，所以需要进行正则化。也就是
                除以n_tokens。
        """

        loss = criteria(out.contiguous().view(-1, out.size(-1)), tgt_y.contiguous().view(-1))
        # 计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()

        total_loss += loss.item()

        # 每40次打印一下loss
        if step != 0 and step % 40 == 0:
            # print("src", src.transpose(0, 1)[0])
            # print("pred", y)
          
          print(f"Step {step}, avg_loss: {total_loss/40:.6f}")
          total_loss = 0.0

def test():
    model.eval()
    # 定义输入序列
    src = torch.LongTensor([[0, 4, 7, 5, 6, 8, 9, 9, 8, 1, 2, 2]])
    # tgt 从 <bos> 开始
    tgt = torch.LongTensor([[0]])

    src = src.transpose(0, 1).to(device)  # (src_len, batch_size)
    tgt = tgt.transpose(0, 1).to(device)  # (1, batch_size)

    with torch.no_grad():
        for i in range(max_length):
            # 送入 transformer
            out = model(src, tgt)  # (tgt_len, batch, d_model)

            # 取最后一个时间步
            last_out = out[-1]          # (batch, d_model)
            predict = model.predictor(last_out)  # (batch, vocab_size)

            # greedy 解码
            y = torch.argmax(predict, dim=-1)    # (batch,)

            # 拼接到 tgt 上
            y = y.unsqueeze(0)   # (1, batch)
            tgt = torch.cat([tgt, y], dim=0)  # (tgt_len+1, batch)

            # 如果预测到 <eos>，提前结束
            if y.item() == 1:
                break

    print("真实结果：", src.transpose(0, 1))
    print("预测结果：", tgt.transpose(0, 1))


if __name__ == '__main__':
    train()
    test()
