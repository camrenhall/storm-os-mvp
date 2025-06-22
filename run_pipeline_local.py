#!/usr/bin/env python3
"""
Local pipeline test - handles requirements and setup automatically
"""
import asyncio
import os
import subprocess
import sys
from pathlib import Path

def check_and_install_requirements():
    """Check if requirements are installed, install if missing"""
    requirements_file = Path('requirements.txt')
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    print("🔍 Checking Python requirements...")
    
    # Try importing key packages to see if requirements are met
    missing_packages = []
    required_packages = ['asyncpg', 'pandas', 'numpy', 'scipy', 'eccodes', 'aiohttp']
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
            ])
            print("✅ Requirements installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install requirements: {e}")
            return False
    else:
        print("✅ All requirements already satisfied")
    
    return True

def create_required_directories():
    """Create cache directories needed by the pipeline"""
    cache_dir = os.getenv('CACHE_DIR', '/tmp/mrms_cache')
    
    directories = [
        cache_dir,
        f"{cache_dir}/flash",
        f"{cache_dir}/qpe", 
        f"{cache_dir}/ffw"
    ]
    
    print(f"📁 Creating cache directories in {cache_dir}...")
    
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")

def load_env_file():
    """Load environment variables from .env file"""
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print("✅ Loaded environment from .env file")
        return True
    else:
        print("⚠️  .env file not found - using system environment variables")
        return False

def validate_environment():
    """Check required environment variables"""
    required_vars = ['DATABASE_URL_DEV']
    missing_vars = []
    
    env = os.getenv('ENV', 'dev')
    if env == 'prod':
        required_vars = ['DATABASE_URL_PROD']
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("💡 Create a .env file with:")
        for var in missing_vars:
            print(f"   {var}=your_database_url_here")
        return False
    
    print("✅ Environment variables validated")
    return True

def main_setup():
    """Run all setup steps before starting pipeline"""
    print("🚀 Starting Flood-Lead Intelligence Pipeline (Local Test)")
    print("=" * 60)
    
    # Step 1: Load environment
    load_env_file()
    
    # Step 2: Validate environment
    if not validate_environment():
        sys.exit(1)
    
    # Step 3: Check and install requirements
    if not check_and_install_requirements():
        sys.exit(1)
    
    # Step 4: Create directories
    create_required_directories()
    
    print("=" * 60)
    print("🎯 Setup complete! Starting pipeline...")
    print("=" * 60)

async def run_pipeline():
    """Import and run the main pipeline"""
    # Import after setup is complete
    from generator.run import main
    await main()

if __name__ == "__main__":
    # Run setup
    main_setup()
    
    # Run pipeline
    asyncio.run(run_pipeline())