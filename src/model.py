import torch
import torch.nn as nn


class StorePredictionModel(nn.Module):
    def __init__(self, num_classes, embed_dim=32, coord_dim=8, lstm_hidden=64, lstm_layers=1, dropout=0.1):
        """
        门店选址预测模型。

        参数:
        num_classes (int): 网格类别的总数。
        embed_dim (int): 网格ID嵌入的维度。
        coord_dim (int): 坐标嵌入的维度。如果为0，则不使用坐标嵌入。
        lstm_hidden (int): LSTM隐藏层的维度。
        lstm_layers (int): LSTM的层数。
        dropout (float): Dropout的比例。
        """
        super(StorePredictionModel, self).__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.coord_dim = coord_dim
        self.lstm_hidden = lstm_hidden

        # 网格ID嵌入层：将每个grid索引映射为embed_dim维向量
        self.id_embedding = nn.Embedding(num_classes, embed_dim)

        # 坐标嵌入：将2维坐标映射为coord_dim维向量
        # 只有当coord_dim > 0 时才创建此层
        if self.coord_dim > 0:
            self.coord_embedding_layer = nn.Linear(2, coord_dim)
        else:
            self.coord_embedding_layer = None

        # LSTM层：输入维度为 embed_dim + coord_dim (如果使用坐标嵌入)
        current_input_dim = embed_dim
        if self.coord_embedding_layer is not None:
            current_input_dim += coord_dim

        self.lstm = nn.LSTM(current_input_dim, lstm_hidden, num_layers=lstm_layers, batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0)
        # 注意: LSTM自带的dropout只在多层时作用于层间，单层时不起作用。
        # 因此，我们在LSTM输出后再加一个独立的Dropout层。

        # Dropout层
        self.dropout_layer = nn.Dropout(dropout)

        # 全连接输出层：映射到num_classes维，用于分类预测
        self.output_fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, seq_ids, seq_coords=None):
        """
        模型的前向传播。

        参数:
        seq_ids (torch.Tensor): 张量 (batch, seq_len)，每个元素为网格的索引表示。
        seq_coords (torch.Tensor, optional): 张量 (batch, seq_len, 2)，对应每个网格的坐标。
                                            如果 coord_embedding_layer 为 None，则此参数被忽略。

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
            lstm_input = torch.cat([id_emb, coord_emb], dim=-1)
        elif self.coord_embedding_layer is not None and seq_coords is None:
            # print("警告: 模型配置了坐标嵌入，但未提供 seq_coords。将仅使用ID嵌入。")
            pass  # lstm_input 保持为 id_emb

        # 3. 通过 LSTM 层
        # output: (batch, seq_len, lstm_hidden) - 所有时间步的输出
        # h_n: (num_layers, batch, lstm_hidden) - 最后一个时间步的隐藏状态
        # c_n: (num_layers, batch, lstm_hidden) - 最后一个时间步的细胞状态
        lstm_output, (h_n, c_n) = self.lstm(lstm_input)

        # 4. 提取序列最后一个时间步的输出作为整体序列表示
        # lstm_output 包含了所有时间步的输出，我们取最后一个
        # last_out: (batch, lstm_hidden)
        last_out = lstm_output[:, -1, :]

        # 5. 应用 dropout
        last_out_dropped = self.dropout_layer(last_out)

        # 6. 输出预测得分 (logits)
        # logits: (batch, num_classes)
        logits = self.output_fc(last_out_dropped)

        return logits


if __name__ == '__main__':
    # 示例用法
    num_classes_example = 100  # 假设有100个不同的网格ID
    embed_dim_example = 32
    coord_dim_example = 8  # 使用坐标嵌入
    lstm_hidden_example = 64

    # 创建模型实例
    model_with_coords = StorePredictionModel(num_classes_example, embed_dim_example, coord_dim_example,
                                             lstm_hidden_example)
    model_no_coords = StorePredictionModel(num_classes_example, embed_dim_example, 0,
                                           lstm_hidden_example)  # coord_dim=0

    # 准备伪输入数据
    batch_size_example = 4
    seq_len_example = 10

    # (batch, seq_len) - 网格ID序列
    dummy_seq_ids = torch.randint(0, num_classes_example, (batch_size_example, seq_len_example))
    # (batch, seq_len, 2) - 对应的坐标序列 (归一化到0-1)
    dummy_seq_coords = torch.rand(batch_size_example, seq_len_example, 2)

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

    print("\n测试带坐标嵌入的模型，但不提供坐标输入 (应有警告或正常运行):")
    try:
        logits_output_wc_no_coords_input = model_with_coords(dummy_seq_ids)  # 故意不传坐标
        print(f"输入ID序列形状: {dummy_seq_ids.shape}")
        print(
            f"输出Logits形状: {logits_output_wc_no_coords_input.shape} (应为: ({batch_size_example}, {num_classes_example}))")
    except Exception as e:
        print(f"模型(配置了坐标嵌入)但不提供坐标输入时出错: {e}")

