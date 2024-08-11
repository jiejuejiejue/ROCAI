import torch
import streamlit as st
import subprocess
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer
import appbuilder 
from transformers import BertModel
#加载预训练模型
pretrained = BertModel.from_pretrained('hfl/chinese-macbert-base')
#需要移动到cuda上
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pretrained.to(device)
#不训练,不需要计算梯度
for param in pretrained.parameters():
    param.requires_grad_(False)
    
#多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        # 确保隐藏层特征数能够被头数整除
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads  # 计算每个头的维度
        # 定义线性层，用于对查询、键、值进行线性变换
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, hidden_size)  # 定义输出线性层，用于整合多头注意力后的输出

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        # 对输入进行线性变换，并将其分割为多个头
        q = self.linear_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.linear_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.linear_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        attn_weights = F.softmax(scores, dim=-1)  # 计算注意力权重
        # 根据注意力权重对值进行加权求和
        context = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        out = self.linear_out(context)  # 整合多头注意力的输出
        return out
    
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(768, 512)  # 第一层全连接层
        self.fc2 = nn.Linear(512, 256)  # 第二层全连接层
        self.fc3 = nn.Linear(256, 2)    # 第三层全连接层
        self.dropout = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.activation = nn.ReLU()
        self.multihead_attention = MultiHeadAttention(hidden_size=768, num_heads=8)  # 多头注意力模块

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = pretrained(input_ids=input_ids,
                         attention_mask=attention_mask,
                         token_type_ids=token_type_ids).last_hidden_state

        # 应用多头注意力机制
        out = self.multihead_attention(out)
        out = out[:, 0]  # 提取[CLS]标记的输出

        out = self.activation(self.bn1(self.fc1(out)))
        out = self.dropout(out)
        out = self.activation(self.bn2(self.fc2(out)))
        out = self.dropout(out)
        out = self.fc3(out)
        out = out.softmax(dim=1)
        return out
    
    
def load_models_and_predict(text, device):
    # 加载模型
    MacBERT_base_CDialBias = torch.load('ROCAI\models\MacBERT-base-CDialBias.pth')
    MacBERT_base_CDialBias.to(device)
    MacBERT_base_COLD = torch.load('ROCAI\models\MacBERT-base-COLD.pth')
    MacBERT_base_COLD.to(device)

    # 获取密钥和ID
    os.environ['APPBUILDER_TOKEN'] = "bce-v3/ALTAK-n2XgeA6FS3Q5E7Jab6UwE/850b44ebec64c4cad705986ab0b5e3df4b05d407"
    app_id = "df881861-9fa6-40b6-b3bd-26df5f5d4b9a"

    # 初始化agent实例
    your_agent = appbuilder.AppBuilderClient(app_id)

    # 创建会话id
    conversation_id = your_agent.create_conversation()

    # 加载字典和分词工具
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-macbert-base')

    # 对输入文本进行编码
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # 将输入数据移动到相同的设备上
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 设置模型为评估模式
    MacBERT_base_CDialBias.eval()
    MacBERT_base_COLD.eval()

    # 调用千帆api获取标签
    msg = your_agent.run(conversation_id, text)
    answer = msg.content.answer

    # 进行预测
    with torch.no_grad():
        out1 = MacBERT_base_CDialBias(**inputs)
    with torch.no_grad():
        out2 = MacBERT_base_COLD(**inputs)

    out1 = torch.argmax(out1, dim=1).item()
    out2 = torch.argmax(out2, dim=1).item()
    out3 = answer[0]

    # 分析结果
    if out3 == "1":
        if out1 == out2 == out3 == 1:
            result = "这句话具有攻击性和社会偏见"
        elif out1 == 0 and out2 == 1:
            result = "这句话具有攻击性，但无社会偏见"
        elif out1 == 1 and out2 == 0:
            result = "这句话不具有攻击性，但具有社会偏见"
        else:
            result = "这句话具有攻击性"
    elif out3 == "0":
        if out1 == out2 == out3 == 0:
            result = "这句话不具有攻击性和社会偏见"
        elif out1 == 0 and out2 == 1:
            result = "这句话具有攻击性，但无社会偏见"
        elif out1 == 1 and out2 == 0:
            result = "这句话不具有攻击性，但具有社会偏见"
        else:
            result = "这句话不具有攻击性"
    return result
def run_command(command):
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")

#创建网页
st.title("☁礼貌用语检测器")

with st.sidebar:
    # 初始化session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    # 用户名和密码输入
    username = st.sidebar.text_input('用户名')
    password = st.sidebar.text_input('密码', type='password')

    # 登录按钮
    if st.sidebar.button('登录'):
    # 这里可以添加验证逻辑，例如检查用户名和密码是否正确
        if username == 'admin' and password == '12345':
            st.session_state.logged_in = True
            st.sidebar.success('登录成功！')
    

        else:
            st.error('用户名或密码错误，请重试。')
            st.stop()

#清空消息
clear = st.button("清除")
if clear:
    st.session_state.clear()

st.divider()

#输出内容
if "memory" not in st.session_state:
    st.session_state['memory'] = []
    st.session_state['message'] = [{"role": "ai",
                                "content": "你好！我是“礼貌用语检测器”。在这里，我能够帮助你检测中文语言中的攻击性和社会偏见内容，维护一个文明、和谐的交流环境。请告诉我你的需求，我会尽力提供帮助。"}]

for message in st.session_state['message']:
    st.chat_message(message["role"]).write(message["content"])

#输入内容
text = st.chat_input()

#运行
if text and st.session_state.logged_in == True:
    #将问题保存进message和memory
    st.session_state["message"].append({"role": "human", "content": text})
    st.session_state["memory"].append(text)
    st.chat_message("human").write(text)
    #得到回答
    with st.spinner("AI正在思考中，请稍等..."):
        result = load_models_and_predict(text, device)

    #将回答保存进message和memory
    st.session_state["message"].append({"role": "ai", "content": result})
    st.session_state["memory"].append(result)
    st.chat_message("ai").write(result)
    
elif text and st.session_state.logged_in == False:
    st.error('请先登录！')
    st.stop()