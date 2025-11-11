import sys
from pathlib import Path

def test_files_exist():
    """Test if all required files exist"""
    print("\n" + "="*60)
    print("🧪 TEST: STREAMLIT FILES")
    print("="*60 + "\n")
    
    required_files = [
        'app.py',
        '.streamlit/config.toml',
        'pages/1_Model_Performance.py',
        'pages/2_Batch_Prediction.py',
        'pages/3_About.py'
    ]
    
    print("Checking required files...")
    
    all_exist = True
    for file in required_files:
        exists = Path(file).exists()
        status = "✅" if exists else "❌"
        print(f"   {status} {file}")
        if not exists:
            all_exist = False
    
    if all_exist:
        print("\n✅ All Streamlit files created!")
    else:
        print("\n❌ Some files missing!")
    
    print("\n" + "="*60)
    print("✅ FILE CHECK COMPLETE!")
    print("="*60 + "\n")
    
    return all_exist


def test_model_exists():
    """Test if trained model exists"""
    print("\n" + "="*60)
    print("🧪 TEST: MODEL AVAILABILITY")
    print("="*60 + "\n")
    
    model_path = Path('models/checkpoints_real/best_model.pth')
    
    if model_path.exists():
        print(f"✅ Model found: {model_path}")
        
        # Check model size
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"   Size: {size_mb:.2f} MB")
        
        status = True
    else:
        print(f"❌ Model not found: {model_path}")
        print("   Please train the model first ")
        status = False
    
    print("\n" + "="*60)
    print("✅ MODEL CHECK COMPLETE!")
    print("="*60 + "\n")
    
    return status


def main():
    """Run all tests"""
    print("\n🚀 STREAMLIT UI TESTS")
    print("Checking files and model availability\n")
    
    try:
        # Test 1: Files
        files_ok = test_files_exist()
        
        # Test 2: Model
        model_ok = test_model_exists()
        
        # Summary
        print("\n" + "="*60)
        print("🎉 ALL TESTS COMPLETED!")
        print("="*60)
        print(f"\n✅ Files: {'PASSED' if files_ok else 'FAILED'}")
        print(f"✅ Model: {'PASSED' if model_ok else 'FAILED'}")
        
        if files_ok and model_ok:
            print("\n🚀 Ready to launch Streamlit app!")
            print("\n📝 To run the app:")
            print("   cd ~/vision-ai-system")
            print("   streamlit run app.py")
            print("\n🌐 App will open at: http://localhost:8501")
        else:
            print("\n⚠️ Please fix issues before running app")
        
        print()
        
        return 0 if (files_ok and model_ok) else 1
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())