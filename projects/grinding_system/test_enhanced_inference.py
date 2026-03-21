"""
增强版 VLA 模型推理测试
====================
测试训练好的增强版模型在 PushT 数据集上的表现
"""

import sys
import torch
from pathlib import Path
import numpy as np
import json

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_enhanced_inference(checkpoint_path: str = None):
    """测试增强版模型推理"""
    print("="*70)
    print("增强版 VLA 模型推理测试")
    print("="*70)
    
    # 默认使用增强版的最佳模型
    if checkpoint_path is None:
        checkpoint_path = Path('outputs/checkpoints_enhanced/best_model_enhanced.pt')
    else:
        checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"❌ 模型文件不存在：{checkpoint_path}")
        print(f"💡 请检查路径是否正确")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print(f"\n✅ 加载模型成功")
    print(f"   路径：{checkpoint_path}")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   最佳像素误差：{checkpoint.get('pixel_error', 'N/A'):.2f} px")
    
    # 显示配置信息
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"\n📋 训练配置:")
        print(f"   数据集：{config.get('dataset_name', 'N/A')}")
        print(f"   Batch size: {config.get('batch_size', 'N/A')}")
        print(f"   学习率：{config.get('learning_rate', 'N/A')}")
        print(f"   训练轮数：{config.get('num_epochs', 'N/A')}")
    
    # 加载数据集
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset('lerobot/pusht')
    
    # 导入增强版模型
    from train_lerobot_vla_enhanced import EnhancedVLAModel
    
    # 获取数据集信息
    sample = dataset[0]
    state_dim = sample['observation.state'].shape[0]
    action_dim = sample['action'].shape[0]
    
    # 创建模型
    model = EnhancedVLAModel(
        state_dim=state_dim,
        action_dim=action_dim
    )
    
    # 加载模型权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    print(f"\n📊 模型信息:")
    print(f"   设备：{device}")
    print(f"   State 维度：{state_dim}")
    print(f"   Action 维度：{action_dim}")
    print(f"   参数量：{sum(p.numel() for p in model.parameters()):,}")
    
    # 测试多个样本
    print(f"\n🧪 推理测试...")
    
    # 检测数据集键名
    sample = dataset[0]
    image_key = None
    possible_image_keys = ['observation.image', 'observation.images.top', 'observation.images.rgb']
    
    for key in possible_image_keys:
        if key in sample:
            image_key = key
            break
    
    if image_key is None:
        for key in sample.keys():
            if 'image' in key.lower():
                image_key = key
                break
    
    state_key = 'observation.state'
    action_key = 'action'
    
    errors = []
    predictions_list = []
    targets_list = []
    pixel_errors = []
    
    # 归一化统计（如果存在）
    norm_stats = checkpoint.get('normalization_stats', None)
    use_normalization = norm_stats is not None
    
    if use_normalization:
        print(f"\n✅ 使用归一化统计")
        action_mean = torch.tensor(norm_stats['action_mean']).to(device)
        action_std = torch.tensor(norm_stats['action_std']).to(device) + 1e-6
    else:
        print(f"\n⚠️  未找到归一化统计，使用原始预测")
    
    for i in range(min(50, len(dataset))):
        sample = dataset[i]
        
        # 准备 batch
        batch = {
            image_key: sample[image_key].unsqueeze(0).to(device),
            state_key: sample[state_key].unsqueeze(0).to(device)
        }
        
        # 推理
        with torch.no_grad():
            prediction = model(batch)
        
        predicted = prediction['actions'].squeeze()
        true_action = torch.tensor(sample[action_key]).to(device)
        
        # 反归一化（如果需要）
        if use_normalization:
            predicted = predicted * action_std + action_mean
        
        predicted_np = predicted.cpu().numpy()
        true_np = true_action.cpu().numpy()
        
        # 计算误差
        mse = np.mean((predicted_np - true_np) ** 2)
        mae = np.mean(np.abs(predicted_np - true_np))
        rmse = np.sqrt(mse)
        pixel_err = np.sqrt(np.sum((predicted_np - true_np) ** 2))
        
        errors.append(mse)
        pixel_errors.append(pixel_err)
        predictions_list.append(predicted_np)
        targets_list.append(true_np)
        
        if i < 10:  # 显示前 10 个样本
            print(f"\n样本 {i+1}:")
            print(f"   预测：{predicted_np}")
            print(f"   真实：{true_np}")
            print(f"   绝对误差：{np.abs(predicted_np - true_np)}")
            print(f"   MSE: {mse:.4f}, RMSE: {rmse:.4f}, Pixel: {pixel_err:.2f}px")
    
    # 整体统计
    all_predictions = np.array(predictions_list)
    all_targets = np.array(targets_list)
    
    avg_mse = np.mean(errors)
    std_mse = np.std(errors)
    overall_mae = np.mean(np.abs(all_predictions - all_targets))
    overall_rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))
    avg_pixel_error = np.mean(pixel_errors)
    
    print(f"\n{'='*70}")
    print(f"📊 总体评估 (测试样本数：{len(errors)})")
    print(f"{'='*70}")
    print(f"   平均 MSE:      {avg_mse:.4f} ± {std_mse:.4f}")
    print(f"   平均 MAE:      {overall_mae:.4f}")
    print(f"   平均 RMSE:     {overall_rmse:.4f}")
    print(f"   平均像素误差： {avg_pixel_error:.2f} px")
    print(f"\n📊 误差分布:")
    print(f"   最小 MSE: {np.min(errors):.4f}")
    print(f"   最大 MSE: {np.max(errors):.4f}")
    print(f"   中位数 MSE: {np.median(errors):.4f}")
    print(f"   像素误差范围：[{np.min(pixel_errors):.2f}, {np.max(pixel_errors):.2f}] px")
    
    # 性能评价
    print(f"\n💡 性能评价:")
    if avg_pixel_error < 5.0:
        print(f"   🏆 优秀！像素误差 < 5px，可以用于实际任务")
    elif avg_pixel_error < 10.0:
        print(f"   ✅ 良好！像素误差 < 10px，达到目标")
    elif avg_pixel_error < 20.0:
        print(f"   ⚠️  可用，但建议继续优化")
    else:
        print(f"   ❌ 需要改进")
    
    # 保存评估结果
    results = {
        'checkpoint_path': str(checkpoint_path),
        'epoch': checkpoint.get('epoch', 'N/A'),
        'dataset': 'lerobot/pusht',
        'n_samples': len(errors),
        'avg_mse': float(avg_mse),
        'std_mse': float(std_mse),
        'mae': float(overall_mae),
        'rmse': float(overall_rmse),
        'avg_pixel_error': float(avg_pixel_error),
        'min_mse': float(np.min(errors)),
        'max_mse': float(np.max(errors)),
        'median_mse': float(np.median(errors)),
        'min_pixel_error': float(np.min(pixel_errors)),
        'max_pixel_error': float(np.max(pixel_errors))
    }
    
    results_path = Path('outputs/checkpoints_enhanced/inference_results.json')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 评估结果已保存：{results_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='测试增强版 VLA 模型')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='模型检查点路径')
    
    args = parser.parse_args()
    
    test_enhanced_inference(args.checkpoint)