"""
Simple colored logger for clean training output.
"""

import sys

class Logger:
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    END = "\033[0m"

    def info(self, msg):
        print(f"{self.BLUE}[INFO]{self.END} {msg}")
        sys.stdout.flush()

    def success(self, msg):
        print(f"{self.GREEN}[OK]{self.END} {msg}")
        sys.stdout.flush()

    def warn(self, msg):
        print(f"{self.YELLOW}[WARN]{self.END} {msg}")
        sys.stdout.flush()

    def error(self, msg):
        print(f"{self.RED}[ERROR]{self.END} {msg}")
        sys.stdout.flush()


logger = Logger()
