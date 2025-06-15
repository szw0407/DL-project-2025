import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import time

class StorePredictionModel(nn.Module):
    def __init__(self, num_classes, embed_dim=32, coord_dim=8, lstm_hidden=128, lstm_layers=3, dropout=0.1, 
                 bert_model_name='bert-base-chinese', use_bert=True, bert_feature_dim=768):
        """
        门店选址预测模型。

        参数:
        num_classes (int): 网格类别的总数。
        embed_dim (int): 网格ID嵌入的维度。
        coord_dim (int): 坐标嵌入的维度。如果为0，则不使用坐标嵌入。
        lstm_hidden (int): LSTM隐藏层的维度。
        lstm_layers (int): LSTM的层数。
        dropout (float): Dropout的比例。        
        bert_model_name (str): 使用的BERT模型名称。
        use_bert (bool): 是否使用BERT特征提取。
        bert_feature_dim (int): BERT特征的维度，用于特征对齐。
        """
        super(StorePredictionModel, self).__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.coord_dim = coord_dim
        self.lstm_hidden = lstm_hidden
        self.use_bert = use_bert
        self.bert_feature_dim = bert_feature_dim

        # BERT模型用于提取店铺名称和类型的语义特征
        if self.use_bert:
            try:
                self.bert_model = BertModel.from_pretrained(bert_model_name)
                self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
                # 冻结BERT参数以减少计算量（可选）
                for param in self.bert_model.parameters():
                    param.requires_grad = False
            except:
                print(f"警告: 无法加载BERT模型 {bert_model_name}，将禁用BERT特征")
                self.use_bert = False
        
        # BERT特征对齐层：将BERT输出特征对齐到序列长度
        if self.use_bert:
            self.bert_feature_projection = nn.Linear(self.bert_feature_dim, embed_dim)
            self.bert_dropout = nn.Dropout(dropout)

        # 网格ID嵌入层：将每个grid索引映射为embed_dim维向量
        self.id_embedding = nn.Embedding(num_classes, embed_dim)

        # 坐标嵌入：将2维坐标映射为coord_dim维向量
        # 只有当coord_dim > 0 时才创建此层
        if self.coord_dim > 0:
            self.coord_embedding_layer = nn.Linear(2, coord_dim)
        else:
            self.coord_embedding_layer = None

        # LSTM层：输入维度为 embed_dim + coord_dim (如果使用坐标嵌入) + embed_dim (如果使用BERT)
        current_input_dim = embed_dim
        if self.coord_embedding_layer is not None:
            current_input_dim += coord_dim
        if self.use_bert:
            current_input_dim += embed_dim  # BERT特征经过投影后的维度

        self.lstm = nn.LSTM(current_input_dim, lstm_hidden, num_layers=lstm_layers, batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0)
        # 注意: LSTM自带的dropout只在多层时作用于层间，单层时不起作用。
        
        # Dropout层
        self.dropout_layer = nn.Dropout(dropout)

        # 复杂的MLP层
        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 全连接输出层：映射到num_classes维，用于分类预测
        self.output_fc = nn.Linear(lstm_hidden // 2, num_classes)

    def extract_bert_features(self, brand_names, brand_types, seq_len):
        """
        从店铺名称和类型中提取BERT特征
        
        参数:
        brand_names (list): 店铺名称列表
        brand_types (list): 店铺类型列表
        seq_len (int): 序列长度，用于特征对齐
        
        返回:
        torch.Tensor: BERT特征张量 (batch, seq_len, embed_dim)
        """
        if not self.use_bert:
            return None
            
        batch_size = len(brand_names)
        device = next(self.parameters()).device
        
        # 将店铺名称和类型组合成文本
        combined_texts = []
        for name, type_info in zip(brand_names, brand_types):
            combined_text = f"{name} {type_info}"
            combined_texts.append(combined_text)
        
        # 使用BERT tokenizer处理文本
        with torch.no_grad():
            encoded = self.bert_tokenizer(
                combined_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            # 将输入移到正确的设备
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            # 获取BERT特征
            bert_outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
            # 使用[CLS] token的表示作为整体文本特征
            bert_features = bert_outputs.last_hidden_state[:, 0, :]  # (batch, bert_feature_dim)
        
        # 投影到指定维度
        projected_features = self.bert_feature_projection(bert_features)  # (batch, embed_dim)
        projected_features = self.bert_dropout(projected_features)
        
        # 扩展到序列长度维度，使每个时间步都有相同的BERT特征
        # (batch, embed_dim) -> (batch, seq_len, embed_dim)
        bert_seq_features = projected_features.unsqueeze(1).expand(-1, seq_len, -1)
        
        return bert_seq_features

    def forward(self, seq_ids, seq_coords=None, brand_names=None, brand_types=None):
        """
        模型的前向传播。

        参数:
        seq_ids (torch.Tensor): 张量 (batch, seq_len)，每个元素为网格的索引表示。
        seq_coords (torch.Tensor, optional): 张量 (batch, seq_len, 2)，对应每个网格的坐标。
                                            如果 coord_embedding_layer 为 None，则此参数被忽略。
        brand_names (list, optional): 店铺名称列表，长度为batch_size。
        brand_types (list, optional): 店铺类型列表，长度为batch_size。

        返回:
        torch.Tensor: logits 张量 (batch, num_classes)。
        """
        batch_size, seq_len = seq_ids.shape[:2]

        # 1. 获取网格ID嵌入表示 (batch, seq_len, embed_dim)
        id_emb = self.id_embedding(seq_ids)

        # 初始化LSTM输入为ID嵌入
        lstm_input = id_emb

        # 2. 如果使用坐标嵌入
        if self.coord_embedding_layer is not None and seq_coords is not None:
            # 检查坐标维度是否正确
            if seq_coords.shape[-1] != 2:
                raise ValueError(f"坐标张量 seq_coords 的最后一维期望是2 (x,y)，但得到的是 {seq_coords.shape[-1]}")

            # 将坐标转换为向量表示并应用非线性激活
            # (batch, seq_len, 2) -> (batch * seq_len, 2)
            coords_flat = seq_coords.reshape(batch_size * seq_len, 2)
            # (batch * seq_len, 2) -> (batch * seq_len, coord_dim)
            coord_emb_flat = self.coord_embedding_layer(coords_flat)
            # 应用激活函数，例如 Tanh
            coord_emb_activated = torch.tanh(coord_emb_flat)
            # (batch * seq_len, coord_dim) -> (batch, seq_len, coord_dim)
            coord_emb = coord_emb_activated.view(batch_size, seq_len, self.coord_dim)

            # 将ID嵌入和坐标嵌入在特征维度拼接
            lstm_input = torch.cat([lstm_input, coord_emb], dim=-1)
        elif self.coord_embedding_layer is not None and seq_coords is None:
            # print("警告: 模型配置了坐标嵌入，但未提供 seq_coords。将仅使用ID嵌入。")
            pass  # lstm_input 保持为 id_emb

        # 3. 如果使用BERT特征
        if self.use_bert and brand_names is not None and brand_types is not None:
            bert_features = self.extract_bert_features(brand_names, brand_types, seq_len)
            if bert_features is not None:
                # 将BERT特征与现有特征拼接
                lstm_input = torch.cat([lstm_input, bert_features], dim=-1)

        # 4. 通过 LSTM 层
        # output: (batch, seq_len, lstm_hidden) - 所有时间步的输出
        # h_n: (num_layers, batch, lstm_hidden) - 最后一个时间步的隐藏状态
        # c_n: (num_layers, batch, lstm_hidden) - 最后一个时间步的细胞状态
        lstm_output, (h_n, c_n) = self.lstm(lstm_input)

        # 5. 提取序列最后一个时间步的输出作为整体序列表示
        # lstm_output 包含了所有时间步的输出，我们取最后一个
        # last_out: (batch, lstm_hidden)
        last_out = lstm_output[:, -1, :]

        # 6. 应用 dropout
        last_out_dropped = self.dropout_layer(last_out)

        # 7. 通过复杂的MLP
        mlp_output = self.mlp(last_out_dropped)

        # 8. 输出预测得分 (logits)
        # logits: (batch, num_classes)
        logits = self.output_fc(mlp_output)

        return logits


if __name__ == '__main__':
    # 检查GPU可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 示例用法
    num_classes_example = 100  # 假设有100个不同的网格ID
    embed_dim_example = 32
    coord_dim_example = 8  # 使用坐标嵌入
    lstm_hidden_example = 64

    # 创建模型实例（禁用BERT以避免依赖问题）
    model_with_coords = StorePredictionModel(
        num_classes_example, embed_dim_example, coord_dim_example,
        lstm_hidden_example, use_bert=False
    ).to(device)
    model_no_coords = StorePredictionModel(
        num_classes_example, embed_dim_example, 0,
        lstm_hidden_example, use_bert=False
    ).to(device)  # coord_dim=0

    # 准备伪输入数据
    batch_size_example = 4
    seq_len_example = 10

    # (batch, seq_len) - 网格ID序列
    dummy_seq_ids = torch.randint(0, num_classes_example, (batch_size_example, seq_len_example)).to(device)
    # (batch, seq_len, 2) - 对应的坐标序列 (归一化到0-1)
    dummy_seq_coords = torch.rand(batch_size_example, seq_len_example, 2).to(device)
    
    # 伪造店铺名称和类型数据
    dummy_brand_names = ["星巴克", "肯德基", "麦当劳", "必胜客"]
    dummy_brand_types = ["餐饮服务;咖啡厅", "餐饮服务;快餐厅", "餐饮服务;快餐厅", "餐饮服务;披萨店"]

    print("测试带坐标嵌入的模型:")
    # 前向传播
    try:
        logits_output_wc = model_with_coords(dummy_seq_ids, dummy_seq_coords)
        print(f"输入ID序列形状: {dummy_seq_ids.shape}")
        print(f"输入坐标序列形状: {dummy_seq_coords.shape}")
        print(f"输出Logits形状: {logits_output_wc.shape} (应为: ({batch_size_example}, {num_classes_example}))")
    except Exception as e:
        print(f"带坐标嵌入的模型前向传播出错: {e}")

    print("\n测试不带坐标嵌入的模型:")
    try:
        logits_output_nc = model_no_coords(dummy_seq_ids)  # 不传入坐标
        print(f"输入ID序列形状: {dummy_seq_ids.shape}")
        print(f"输出Logits形状: {logits_output_nc.shape} (应为: ({batch_size_example}, {num_classes_example}))")
    except Exception as e:
        print(f"不带坐标嵌入的模型前向传播出错: {e}")

    print("\n测试带BERT特征的模型:")
    try:
        model_with_bert = StorePredictionModel(
            num_classes_example, embed_dim_example, coord_dim_example,
            lstm_hidden_example, use_bert=True
        ).to(device)
        
        # 将输入数据移到GPU（如果使用GPU）
        seq_ids_gpu = dummy_seq_ids
        seq_coords_gpu = dummy_seq_coords
        
        logits_output_bert = model_with_bert(
            seq_ids_gpu, seq_coords_gpu, 
            dummy_brand_names, dummy_brand_types
        )
        print(f"输入ID序列形状: {seq_ids_gpu.shape} (设备: {seq_ids_gpu.device})")
        print(f"输入坐标序列形状: {seq_coords_gpu.shape} (设备: {seq_coords_gpu.device})")
        print(f"品牌名称: {dummy_brand_names}")
        print(f"品牌类型: {dummy_brand_types}")
        print(f"输出Logits形状: {logits_output_bert.shape} (设备: {logits_output_bert.device})")
        print(f"应为: ({batch_size_example}, {num_classes_example})")
    except Exception as e:
        print(f"带BERT特征的模型前向传播出错: {e}")
    
    # 测试性能
    print(f"\n性能测试 (设备: {device}):")
    
    # 预热
    for _ in range(3):
        with torch.no_grad():
            _ = model_with_coords(dummy_seq_ids, dummy_seq_coords)
    
    # 实际测试
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    num_iterations = 100
    with torch.no_grad():
        for _ in range(num_iterations):
            logits = model_with_coords(dummy_seq_ids, dummy_seq_coords)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations * 1000  # 转换为毫秒
    print(f"平均推理时间: {avg_time:.2f} ms")
    print(f"吞吐量: {batch_size_example * num_iterations / (end_time - start_time):.2f} samples/sec")

