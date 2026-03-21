"""
VLA 模型预测结果可视化
====================
可视化模型的预测动作与真实动作的对比
"""

import sys
import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

# 设置中文字体（快速修复版）
def set_chinese_font():
    """设置中文字体 - 快速修复版"""
    try:
        # 尝试设置中文字体
        plt.rcParams['font.family'] = 'SimHei'  # 黑体
        plt.rcParams['axes.unicode_minus'] = False  # 显示负号
        print("✅ 已设置中文字体: SimHei")
    except Exception as e:
        print(f"❌ 设置中文字体时出错: {e}")
        print("💡 将使用默认字体")

# 调用函数
set_chinese_font()
matplotlib.use('Agg')  # 非交互式后端

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def visualize_predictions(checkpoint_path: str = None, save_dir: str = None):
    """可视化模型预测结果"""
    print("="*70)
    print("VLA 模型预测结果可视化")
    print("="*70)
    
    # 默认使用增强版的最佳模型
    if checkpoint_path is None:
        checkpoint_path = Path('outputs/checkpoints_enhanced/best_model_enhanced.pt')
    else:
        checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"❌ 模型文件不存在：{checkpoint_path}")
        return
    
    # 设置保存目录
    if save_dir is None:
        save_dir = Path('outputs/visualizations')
    else:
        save_dir = Path(save_dir)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型和数据集
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from train_lerobot_vla_enhanced import EnhancedVLAModel
    
    dataset = LeRobotDataset('lerobot/pusht')
    
    sample = dataset[0]
    state_dim = sample['observation.state'].shape[0]
    action_dim = sample['action'].shape[0]
    
    model = EnhancedVLAModel(state_dim=state_dim, action_dim=action_dim)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # 归一化统计
    norm_stats = checkpoint.get('normalization_stats', None)
    use_normalization = norm_stats is not None
    
    if use_normalization:
        action_mean = torch.tensor(norm_stats['action_mean']).to(device)
        action_std = torch.tensor(norm_stats['action_std']).to(device) + 1e-6
    
    # 检测键名
    sample = dataset[0]
    image_key = None
    for key in ['observation.image', 'observation.images.top', 'observation.images.rgb']:
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
    
    print(f"\n🧪 测试数据集...")
    
    # 收集预测和真实值
    n_samples = min(100, len(dataset))
    predictions = []
    targets = []
    errors = []
    images = []
    
    for i in range(n_samples):
        sample = dataset[i]
        
        batch = {
            image_key: sample[image_key].unsqueeze(0).to(device),
            state_key: sample[state_key].unsqueeze(0).to(device)
        }
        
        with torch.no_grad():
            prediction = model(batch)
        
        pred = prediction['actions'].squeeze()
        true = torch.tensor(sample[action_key]).to(device)
        
        if use_normalization:
            pred = pred * action_std + action_mean
        
        pred_np = pred.cpu().numpy()
        true_np = true.cpu().numpy()
        
        predictions.append(pred_np)
        targets.append(true_np)
        errors.append(np.abs(pred_np - true_np))
        
        # 保存图像用于可视化
        if image_key in sample:
            img = sample[image_key].permute(1, 2, 0).cpu().numpy()
            images.append(img)
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    errors = np.array(errors)
    
    print(f"✅ 完成 {n_samples} 个样本的预测")
    
    # ========== 开始可视化 ==========
    print(f"\n📊 生成可视化图表...")
    
    # 1. 预测 vs 真实值散点图
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for dim in range(action_dim):
        ax = axes[dim]
        ax.scatter(targets[:, dim], predictions[:, dim], alpha=0.6, s=20)
        ax.plot([targets[:, dim].min(), targets[:, dim].max()], 
                [targets[:, dim].min(), targets[:, dim].max()], 
                'r--', linewidth=2, label='理想预测')
        ax.set_xlabel(f'真实值 (Dim {dim})', fontsize=12)
        ax.set_ylabel(f'预测值 (Dim {dim})', fontsize=12)
        ax.set_title(f'动作维度 {dim}: 预测 vs 真实', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    scatter_path = save_dir / 'predictions_scatter.png'
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 散点图已保存：{scatter_path}")
    
    # 2. 误差分布直方图
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    mse_per_sample = np.mean(errors**2, axis=1)
    axes[0].hist(mse_per_sample, bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(mse_per_sample), color='r', linestyle='--', 
                   label=f'平均 MSE: {np.mean(mse_per_sample):.4f}')
    axes[0].set_xlabel('MSE', fontsize=12)
    axes[0].set_ylabel('样本数', fontsize=12)
    axes[0].set_title('MSE 误差分布', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    pixel_errors = np.sqrt(np.sum(errors**2, axis=1))
    axes[1].hist(pixel_errors, bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[1].axvline(np.mean(pixel_errors), color='r', linestyle='--',
                   label=f'平均像素误差：{np.mean(pixel_errors):.2f}px')
    axes[1].set_xlabel('像素误差 (px)', fontsize=12)
    axes[1].set_ylabel('样本数', fontsize=12)
    axes[1].set_title('像素误差分布', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    histogram_path = save_dir / 'error_histograms.png'
    plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 误差分布图已保存：{histogram_path}")
    
    # 3. 预测轨迹对比图（前 20 个样本）
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()
    
    for i in range(min(20, n_samples)):
        ax = axes[i]
        
        # 绘制真实和预测的动作向量
        origin = [0], [0]
        ax.quiver(*origin, targets[i, 0], targets[i, 1], 
                 color='blue', scale=1, scale_units='xy', angles='xy',
                 label='真实', width=0.005)
        ax.quiver(*origin, predictions[i, 0], predictions[i, 1], 
                 color='red', scale=1, scale_units='xy', angles='xy',
                 label='预测' if i == 0 else "", width=0.005)
        
        # 连接起点和终点
        ax.plot([0, targets[i, 0]], [0, targets[i, 1]], 'b-', alpha=0.3, linewidth=2)
        ax.plot([0, predictions[i, 0]], [0, predictions[i, 1]], 'r--', alpha=0.5, linewidth=2)
        
        error = np.sqrt(np.sum((predictions[i] - targets[i])**2))
        ax.set_title(f'样本 {i+1}\n误差：{error:.2f}px', fontsize=10)
        ax.set_xlim(min(targets[i, 0], predictions[i, 0]) - 2, 
                   max(targets[i, 0], predictions[i, 0]) + 2)
        ax.set_ylim(min(targets[i, 1], predictions[i, 1]) - 2, 
                   max(targets[i, 1], predictions[i, 1]) + 2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(loc='upper right')
    
    plt.tight_layout()
    trajectory_path = save_dir / 'trajectory_comparison.png'
    plt.savefig(trajectory_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 轨迹对比图已保存：{trajectory_path}")
    
    # 4. 误差随样本变化图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sample_indices = range(1, n_samples + 1)
    ax.plot(sample_indices, pixel_errors, 'o-', markersize=3, alpha=0.6, label='像素误差')
    ax.axhline(np.mean(pixel_errors), color='r', linestyle='--', 
              label=f'平均：{np.mean(pixel_errors):.2f}px')
    ax.fill_between(sample_indices, 
                    np.mean(pixel_errors) - np.std(pixel_errors),
                    np.mean(pixel_errors) + np.std(pixel_errors),
                    alpha=0.3, color='gray', label=f'±σ: {np.std(pixel_errors):.2f}')
    
    ax.set_xlabel('样本编号', fontsize=12)
    ax.set_ylabel('像素误差 (px)', fontsize=12)
    ax.set_title('每个样本的预测误差', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    error_trend_path = save_dir / 'error_trend.png'
    plt.savefig(error_trend_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 误差趋势图已保存：{error_trend_path}")
    
    # 5. 带图像的可视化（前 9 个样本）
    if len(images) > 0:
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.flatten()
        
        for i in range(min(9, len(images))):
            ax = axes[i]
            
            # 显示图像
            if images[i].shape[-1] == 3:
                ax.imshow(images[i])
            else:
                ax.imshow(images[i], cmap='gray')
            
            # 叠加预测和真实箭头
            height, width = images[i].shape[:2]
            center_x, center_y = width // 2, height // 2
            
            scale = 50  # 箭头缩放因子
            ax.arrow(center_x, center_y, 
                    predictions[i, 0] * scale, predictions[i, 1] * scale,
                    color='red', width=3, head_width=10, 
                    label='预测', alpha=0.8)
            ax.arrow(center_x, center_y, 
                    targets[i, 0] * scale, targets[i, 1] * scale,
                    color='blue', width=3, head_width=10, 
                    label='真实' if i == 0 else "", alpha=0.8)
            
            error = np.sqrt(np.sum((predictions[i] - targets[i])**2))
            ax.set_title(f'样本 {i+1}\n误差：{error:.2f}px', fontsize=12, color='white')
            ax.axis('off')
            
            if i == 0:
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor='red', alpha=0.8, label='预测'),
                                 Patch(facecolor='blue', alpha=0.8, label='真实')]
                ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        image_viz_path = save_dir / 'predictions_with_images.png'
        plt.savefig(image_viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ 图像可视化已保存：{image_viz_path}")
    
    # 6. 生成综合报告
    report_data = {
        'checkpoint': str(checkpoint_path),
        'n_samples': n_samples,
        'metrics': {
            'avg_mse': float(np.mean(np.mean(errors**2, axis=1))),
            'avg_mae': float(np.mean(errors)),
            'avg_rmse': float(np.sqrt(np.mean(errors**2))),
            'avg_pixel_error': float(np.mean(pixel_errors)),
            'std_pixel_error': float(np.std(pixel_errors)),
            'min_pixel_error': float(np.min(pixel_errors)),
            'max_pixel_error': float(np.max(pixel_errors))
        },
        'visualizations': {
            'scatter_plot': str(scatter_path),
            'error_histograms': str(histogram_path),
            'trajectory_comparison': str(trajectory_path),
            'error_trend': str(error_trend_path)
        }
    }
    
    if len(images) > 0:
        report_data['visualizations']['image_visualization'] = str(image_viz_path)
    
    import json
    report_path = save_dir / 'visualization_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 可视化报告已保存：{report_path}")
    print(f"\n{'='*70}")
    print(f"📊 可视化完成！")
    print(f"{'='*70}")
    print(f"生成的图表:")
    print(f"   1. 预测 vs 真实散点图：{scatter_path}")
    print(f"   2. 误差分布直方图：{histogram_path}")
    print(f"   3. 轨迹对比图：{trajectory_path}")
    print(f"   4. 误差趋势图：{error_trend_path}")
    if len(images) > 0:
        print(f"   5. 图像可视化：{image_viz_path}")
    print(f"   6. 综合报告：{report_path}")
    print(f"\n💡 提示：可以使用图片查看器打开这些 PNG 文件查看可视化结果")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='可视化 VLA 模型预测结果')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='模型检查点路径')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='可视化结果保存目录')
    
    args = parser.parse_args()
    
    visualize_predictions(args.checkpoint, args.save_dir)