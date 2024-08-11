import pandas as pd
import streamlit as st
import torch
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
        
# 页面配置
st.set_page_config(page_title="文件式文本检测工具")
st.title("批量检测攻击性和偏见")

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
    st.divider()
        
# 文件上传
file = st.file_uploader("上传你的CSV文件", type=["csv"])

if file is not None:
    # 读取文件
    df = pd.read_csv(file)
    st.dataframe(df)

    # 输入列名
    column = st.text_input("请输入需要判断的内容的列名")
    
    # 添加保存结果的选项
    save_results = st.checkbox("保存结果为CSV文件")
    
    if st.button("开始检测"):
        if column not in df.columns:
            st.error(f"列名 '{column}' 不存在于数据集中，请检查并重新输入。")
        else:
            # 创建一个新的DataFrame来存储结果
            results_df = pd.DataFrame(columns=['检测文本', '检测结果'])
            
            # 显示进度条
            progress_bar = st.progress(0)
            
            # 初始化停止标志
            stop_flag = False
            
            # 添加停止按钮
            stop_button = st.button("停止检测")
            
            for i, (index, row) in enumerate(df.iterrows()):
                
                # 如果用户点击了停止按钮
                if stop_button:
                    stop_flag = True
                    break
                # 获取特定列的内容
                text = row[column]
                
                # 进行预测
                with st.spinner("AI正在思考中，请稍等..."):
                    result = load_models_and_predict(text, device)
                
                # 将结果添加到新的DataFrame中
                results_df.loc[i] = [text, result]
                r = results_df.loc[i]
 
                # 显示结果
                st.dataframe(r)
                
                st.divider()
                
                # 更新进度条
                progress_bar.progress((i + 1) / len(df))
            
            # 完成处理
            progress_bar.empty()
            
            # 如果用户点击了停止按钮
            if stop_flag:
                st.warning("检测已停止。")
            else:
                st.success("所有文本已检测完成！")
            
            # 如果用户选择了保存结果
            if (save_results and not stop_flag) or st.button("保存结果为CSV文件"):
                # 提供下载链接
                csv_result = results_df.to_csv(index=False)
                st.download_button(
                    label="下载结果",
                    data=csv_result,
                    file_name='results.csv',
                    mime='text/csv'
                )
    
   