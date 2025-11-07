import sys
import torch
import json
from pathlib import Path
import time

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from clip_flowmatching.datasets import VisiumHDDataset
from clip_flowmatching.config import TrainingConfig, AlignerConfig, FlowMatchingConfig, DecoderConfig
from clip_flowmatching.trainer import MultiOmicsToHETrainer, build_training_state

def detailed_train():
    """详细的训练函数，包含完整的错误处理"""
    print("=== 开始详细训练测试 ===")
    start_time = time.time()
    
    try:
        # 设置参数
        data_path = Path("clip_flowmatching/data/Visium_HD_FF")
        cancer_types = ["COAD"]  # 先从一个癌症类型开始
        omics_type = "transcriptome"
        image_size = 128
        device = "cpu"
        
        print(f"数据路径: {data_path}")
        print(f"癌症类型: {cancer_types}")
        print(f"组学类型: {omics_type}")
        print(f"图像大小: {image_size}")
        print(f"设备: {device}")
        
        # 检查数据路径是否存在
        if not data_path.exists():
            raise FileNotFoundError(f"数据路径不存在: {data_path}")
        
        print("\n1. 加载数据集...")
        dataset_start = time.time()
        
        # 尝试加载数据集
        dataset = VisiumHDDataset(
            data_root=data_path,
            cancer_types=cancer_types,
            omics_type=omics_type,
            image_size=image_size,
            max_genes=1000  # 限制基因数量以加快处理
        )
        
        dataset_time = time.time() - dataset_start
        print(f"数据集加载完成，耗时: {dataset_time:.2f}秒")
        print(f"数据集大小: {len(dataset)}")
        
        if len(dataset) == 0:
            raise ValueError("数据集为空")
        
        # 检查样本
        print("\n2. 检查样本...")
        sample = dataset[0]
        print(f"omics形状: {sample.omics.shape}")
        print(f"图像形状: {sample.image.shape}")
        print(f"omics值范围: [{sample.omics.min():.3f}, {sample.omics.max():.3f}]")
        print(f"图像值范围: [{sample.image.min():.3f}, {sample.image.max():.3f}]")
        
        # 创建配置
        print("\n3. 创建配置...")
        omics_dim = sample.omics.shape[0]
        
        aligner_cfg = AlignerConfig(
            omics_dim=omics_dim,
            projection_dim=512,
            temperature=0.07,
            hidden_dims=[512, 256],
            use_batch_norm=False,
            dropout=0.1
        )
        
        flow_cfg = FlowMatchingConfig(
            embedding_dim=512,
            num_layers=4,
            num_heads=4,
            mlp_ratio=4.0,
            dropout=0.1,
            time_embedding_dim=64,
            conditioning_dropout=0.0,
            noise_schedule=[0.0, 0.5, 1.0]
        )
        
        decoder_cfg = DecoderConfig(
            image_size=image_size,
            base_channels=32,
            channel_mults=[1, 2],
            num_res_blocks=1,
            attention_resolutions=[],
            dropout=0.1,
            clip_dim=512
        )
        
        train_cfg = TrainingConfig(
            batch_size=1,  # 小批次
            num_workers=0,  # 避免多进程问题
            max_epochs=2,  # 少量epoch
            precision="fp32",
            aligner_lr=1e-4,
            flow_matching_lr=1e-4,
            decoder_lr=1e-4,
            weight_decay=1e-2,
            alpha=0.5,
            gradient_clip_norm=1.0,
            log_every_n_steps=1,
            eval_every_n_epochs=1
        )
        
        print("配置创建完成")
        
        # 创建设备
        print("\n4. 创建模型...")
        device_obj = torch.device(device)
        
        model_start = time.time()
        training_state = build_training_state(
            aligner_config=aligner_cfg,
            flow_config=flow_cfg,
            decoder_config=decoder_cfg,
            device=device_obj
        )
        model_time = time.time() - model_start
        print(f"模型创建完成，耗时: {model_time:.2f}秒")
        
        # 创建训练器
        trainer = MultiOmicsToHETrainer(
            aligner=training_state.aligner,
            flow_model=training_state.flow_model,
            decoder=training_state.decoder,
            config=train_cfg,
            device=device_obj
        )
        
        # 创建数据加载器
        print("\n5. 创建数据加载器...")
        dataloader = MultiOmicsToHETrainer.build_dataloader(
            dataset,
            batch_size=train_cfg.batch_size,
            num_workers=train_cfg.num_workers
        )
        
        print(f"数据加载器创建完成，批次数量: {len(dataloader)}")
        
        # 训练各个组件
        print("\n6. 开始训练...")
        train_start = time.time()
        
        print("6a. 训练Aligner...")
        trainer.train_aligner(dataloader)
        
        print("6b. 训练Flow Matching...")
        trainer.train_flow_matching(dataloader)
        
        print("6c. 训练Decoder...")
        trainer.train_decoder(dataloader)
        
        train_time = time.time() - train_start
        print(f"训练完成，耗时: {train_time:.2f}秒")
        
        # 保存模型
        print("\n7. 保存模型...")
        save_start = time.time()
        
        torch.save({
            'aligner_state_dict': training_state.aligner.state_dict(),
            'flow_model_state_dict': training_state.flow_model.state_dict(),
            'decoder_state_dict': training_state.decoder.state_dict(),
            'config': {
                'aligner': aligner_cfg,
                'flow_matching': flow_cfg,
                'decoder': decoder_cfg,
                'training': train_cfg
            }
        }, 'trained_model.pth')
        
        save_time = time.time() - save_start
        print(f"模型保存完成，耗时: {save_time:.2f}秒")
        
        total_time = time.time() - start_time
        print(f"\n=== 训练测试成功完成 ===")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"模型已保存到: trained_model.pth")
        
    except Exception as e:
        print(f"\n训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = detailed_train()
    if success:
        print("\n✅ 训练测试成功！")
    else:
        print("\n❌ 训练测试失败！")