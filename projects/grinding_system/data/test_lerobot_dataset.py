"""
测试 LeRobot 数据集
====================
直接使用 LeRobot 库加载标准数据集
支持多种数据集选项
"""

import os


# 可用的公开数据集列表
AVAILABLE_DATASETS = {
    '1': {
        'name': 'PushT',
        'repo_id': 'lerobot/pusht',
        'requires_auth': False,
        'description': 'Tactile pushing dataset (小，适合快速测试)'
    },
    '2': {
        'name': 'ALOHAMobile',
        'repo_id': 'lerobot/aloha_mobile_cabinet',
        'requires_auth': False,
        'description': '移动操作数据集'
    },
    '3': {
        'name': 'Bridge',
        'repo_id': 'lerobot/bridge',
        'requires_auth': True,  # 需要认证
        'description': 'UC Berkeley 桥接数据集 (需要 HF Token)'
    }
}


def setup_huggingface():
    """设置 HuggingFace 认证"""
    print("\n🔐 HuggingFace 设置")
    print("-" * 70)
    
    # 检查是否有 token
    hf_token = os.getenv('HF_TOKEN')
    
    if not hf_token:
        try:
            from huggingface_hub import HfFolder
            hf_token = HfFolder.get_token()
        except:
            hf_token = None
    
    if hf_token:
        print("✅ 检测到已保存的 Token")
        return hf_token
    
    # 询问是否需要输入
    print("ℹ️  部分数据集需要 HuggingFace Token")
    choice = input("\n是否现在设置 Token? (y/n): ").strip().lower()
    
    if choice == 'y':
        print("\n💡 获取 Token 步骤:")
        print("  1. 访问：https://huggingface.co/settings/tokens")
        print("  2. 登录后点击 'New token'")
        print("  3. 选择 'Read' 权限")
        print("  4. 复制生成的 token")
        
        hf_token = input("\n请输入 Token (直接回车跳过): ").strip()
        
        if hf_token:
            from huggingface_hub import login, HfFolder
            HfFolder.save_token(hf_token)
            login(token=hf_token)
            print("✅ Token 已保存并登录")
            return hf_token
        else:
            print("ℹ️  跳过 Token 设置（只能访问公开数据集）")
            return None
    else:
        print("ℹ️  跳过 Token 设置")
        return None


def load_dataset(repo_id: str, hf_token=None):
    """加载数据集"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    
    print(f"\n📥 正在加载数据集：{repo_id}")
    
    try:
        # 尝试加载
        if hf_token:
            dataset = LeRobotDataset(repo_id, token=hf_token)
        else:
            dataset = LeRobotDataset(repo_id)
        
        return dataset, None
        
    except Exception as e:
        error_msg = str(e)
        
        # 如果是认证错误
        if '401' in error_msg or 'authentication' in error_msg.lower():
            print(f"⚠️  此数据集需要认证")
            
            # 尝试重新认证
            print("\n💡 需要 HuggingFace Token")
            new_token = input("请输入 Token (或回车取消): ").strip()
            
            if new_token:
                from huggingface_hub import login, HfFolder
                HfFolder.save_token(new_token)
                login(token=new_token)
                
                # 重试
                print("\n🔄 使用新 Token 重试...")
                try:
                    dataset = LeRobotDataset(repo_id, token=new_token)
                    return dataset, new_token
                except Exception as retry_e:
                    return None, f"重试失败：{retry_e}"
        
        return None, str(e)


def test_lerobot_dataset():
    """测试 LeRobot 数据集"""
    print("="*70)
    print("测试 LeRobot 数据集")
    print("="*70)
    
    # 1. 设置认证
    hf_token = setup_huggingface()
    
    # 2. 显示可用数据集
    print("\n📊 可用的数据集:")
    print("-" * 70)
    for key, info in AVAILABLE_DATASETS.items():
        auth_flag = "🔒" if info['requires_auth'] else "🔓"
        print(f"  {key}. {auth_flag} {info['name']} - {info['description']}")
    
    # 3. 选择数据集
    choice = input("\n选择数据集 (1-3): ").strip()
    
    if choice not in AVAILABLE_DATASETS:
        print("❌ 无效选择")
        return None, None
    
    dataset_info = AVAILABLE_DATASETS[choice]
    repo_id = dataset_info['repo_id']
    
    # 4. 加载数据集
    dataset, result = load_dataset(repo_id, hf_token)
    
    if dataset is None:
        print(f"\n❌ 加载失败：{result}")
        return None, None
    
    # 5. 显示信息
    print(f"\n✅ 数据集加载成功!")
    print(f"   名称：{dataset_info['name']}")
    print(f"   Repo ID: {repo_id}")
    print(f"   大小：{len(dataset)} episodes")
    
    if hasattr(dataset, 'features'):
        print(f"   特征列表:")
        for key in dataset.features:
            print(f"     - {key}")
    
    # 6. 查看样本
    print("\n📊 查看第一个样本...")
    try:
        sample = dataset[0]
        
        print(f"   样本键:")
        for key, value in sample.items():
            if hasattr(value, 'shape'):
                print(f"     - {key}: {value.shape}")
            else:
                print(f"     - {key}: {type(value).__name__}")
        
        return dataset, sample
        
    except Exception as e:
        print(f"⚠️  无法查看样本：{e}")
        return dataset, None


def main():
    """主函数"""
    dataset, sample = test_lerobot_dataset()
    
    if dataset is not None:
        print("\n" + "="*70)
        print("✅ LeRobot 数据集测试成功!")
        print("="*70)
        
        # 获取选择的数据集信息
        choice = input("\n刚才选择的数据集编号 (1-3): ").strip()
        dataset_info = AVAILABLE_DATASETS.get(choice, {})
        
        print("\n下一步:")
        print("  1. 使用此数据集训练VLA 模型")
        print(f"     python projects/grinding_system/train_lerobot_vla.py")
        if dataset_info:
            print(f"     (数据集：{dataset_info['repo_id']})")
        print("\n  2. 修改训练配置")
        print("     编辑 train_lerobot_vla.py 中的 config 字典")

if __name__ == "__main__":
    main()