"""让 pytest 能从仓库根导入 `src` 包。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
