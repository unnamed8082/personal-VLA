"""
VLAb 数据集下载器
==================
支持从 HuggingFace Hub 下载 VLAb 社区数据集
"""

from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
from pathlib import Path
import os


def setup_proxy():
    """设置代理服务器"""
    print("="*70)
    print("设置网络代理")
    print("="*70)
    
    # 使用用户提供的代理配置
    proxy_url = "http://127.0.0.1:7897"
    
    os.environ['HTTP_PROXY'] = proxy_url
    os.environ['HTTPS_PROXY'] = proxy_url
    os.environ['http_proxy'] = proxy_url
    os.environ['https_proxy'] = proxy_url
    
    print(f"\n✅ 代理设置成功:")
    print(f"   HTTP_PROXY: {proxy_url}")
    print(f"   HTTPS_PROXY: {proxy_url}")
    
    # 验证网络连接
    try:
        import requests
        response = requests.get('https://huggingface.co', timeout=5)
        print(f"   ✅ 网络连接正常 (状态码：{response.status_code})")
    except Exception as e:
        print(f"   ⚠️  网络连接测试失败：{e}")
        print(f"   💡 请确保代理服务器正在运行")


def download_vlab_dataset(
    repo_id: str = "VLA-Bench/vlab-community-v1",
    save_dir: str = "datasets/vlab_v1",
    subset: str = None,
    max_episodes: int = None
):
    """
    下载 VLAb 数据集
    
    Args:
        repo_id: HuggingFace 仓库 ID
        save_dir: 保存目录
        subset: 子集名称（可选）
        max_episodes: 最大下载 episode 数（用于快速测试）
    """
    print("="*70)
    print("VLAb 数据集下载")
    print("="*70)
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📥 正在下载数据集:")
    print(f"   仓库：{repo_id}")
    print(f"   保存路径：{save_path}")
    
    try:
        # 方法 1: 下载整个仓库（推荐小规模数据集）
        if max_episodes is None or max_episodes > 100:
            print("\n🔄 使用 snapshot_download 下载完整数据集...")
            
            downloaded_path = snapshot_download(
                repo_id=repo_id,
                local_dir=str(save_path),
                local_dir_use_symlinks=False,  # Windows 兼容性
                resume_download=True,
                proxies={
                    'http': 'http://127.0.0.1:7897',
                    'https': 'http://127.0.0.1:7897'
                }
            )
            
            print(f"✅ 数据集已下载到：{downloaded_path}")
        
        # 方法 2: 只下载部分文件（用于快速测试）
        else:
            print(f"\n🔄 仅下载前 {max_episodes} 个 episode...")
            
            # 列出所有文件
            files = list_repo_files(repo_id)
            
            # 过滤出需要的文件
            episodes_to_download = []
            for i in range(max_episodes):
                ep_prefix = f"episode_{i}"
                episode_files = [f for f in files if f.startswith(ep_prefix)]
                episodes_to_download.extend(episode_files)
            
            if len(episodes_to_download) == 0:
                print("⚠️  未找到匹配的 episode 文件，检查仓库结构...")
                # 显示前几个文件作为调试
                print(f"   仓库中的前 20 个文件:")
                for f in files[:20]:
                    print(f"     - {f}")
                return None
            
            # 逐个下载
            for i, filename in enumerate(episodes_to_download):
                print(f"\r下载进度：{i+1}/{len(episodes_to_download)}", end='')
                
                file_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=str(save_path),
                    local_dir_use_symlinks=False,
                    proxies={
                        'http': 'http://127.0.0.1:7897',
                        'https': 'http://127.0.0.1:7897'
                    }
                )
            
            print(f"\n✅ 已下载 {max_episodes} 个 episode")
        
        # 验证下载
        print("\n🔍 验证下载完整性...")
        n_episodes = len(list(save_path.glob("episode_*")))
        print(f"   下载的 episode 数：{n_episodes}")
        
        if n_episodes == 0:
            print("⚠️  未找到 episode 目录，检查数据结构...")
            # 列出目录内容
            items = list(save_path.iterdir())[:10]
            print(f"   前 10 个项目:")
            for item in items:
                print(f"     - {item.name}")
        
        print(f"\n💾 数据集位置：{save_path.absolute()}")
        
    except Exception as e:
        print(f"\n❌ 下载失败：{e}")
        print("\n💡 提示:")
        print("  1. 确认代理服务器是否运行 (端口 7897)")
        print("  2. 检查网络连接")
        print("  3. 尝试减少下载的 episode 数量")
        raise
    
    return save_path


def setup_huggingface_token():
    """设置 HuggingFace Token（可选，用于加速下载）"""
    print("="*70)
    print("设置 HuggingFace Token")
    print("="*70)
    
    token = input("\n请输入你的 HuggingFace Token (可选，直接回车跳过): ")
    
    if token.strip():
        from huggingface_hub import login
        login(token=token)
        print("✅ Token 设置成功")
    else:
        print("ℹ️  跳过 Token 设置（下载速度可能较慢）")


def main():
    """主函数"""
    print("\n🚀 VLAb 数据集下载工具")
    print("="*70)
    
    # 1. 先设置代理
    setup_proxy()
    
    # 2. 选择下载模式
    print("\n选择下载模式:")
    print("1. 快速测试 (下载 10 个 episode)")
    print("2. 完整下载 (下载全部 11.1K episodes)")
    print("3. 自定义数量")
    
    choice = input("\n输入选项 (1/2/3): ")
    
    if choice == '1':
        max_eps = 10
        print(f"\n📦 快速测试模式：下载 {max_eps} 个 episode")
    elif choice == '2':
        max_eps = None
        print("\n📦 完整下载模式：下载全部 episodes")
    else:
        max_eps = int(input("输入要下载的 episode 数量: "))
        print(f"\n📦 自定义模式：下载 {max_eps} 个 episode")
    
    # 3. 设置 Token（可选）
    setup_huggingface_token()
    
    # 4. 开始下载
    try:
        dataset_path = download_vlab_dataset(
            repo_id="VLA-Bench/vlab-community-v1",
            save_dir="datasets/vlab_v1",
            max_episodes=max_eps
        )
        
        if dataset_path is None:
            print("\n⚠️  下载可能不完整，请检查输出信息")
            return
        
        print("\n" + "="*70)
        print("✅ 数据集下载完成！")
        print("="*70)
        print(f"\n下一步:")
        print(f"  1. 运行数据集测试验证:")
        print(f"     python projects/grinding_system/data/vlab_dataset.py")
        print(f"\n  2. 开始训练 VLA 模型:")
        print(f"     python projects/grinding_system/train_vla.py")
        
    except Exception as e:
        print(f"\n❌ 下载过程中断：{e}")
        print("\n💡 可以尝试:")
        print("  1. 确认代理服务器是否运行在 127.0.0.1:7897")
        print("  2. 重启代理软件")
        print("  3. 减少下载的 episode 数量")


if __name__ == "__main__":
    main()