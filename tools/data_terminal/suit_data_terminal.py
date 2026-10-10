"""SUIT Data Terminal: one double-click opens an SSH terminal on the OpenMediaVault data machine, already
inside the solar DATA folder (the folder the \\\\192.168.1.2\\DATA share shows).
"""
import argparse
import os
import shutil
import subprocess
import sys

HOST = "192.168.1.2"
USER = "aniruddha"
DATA = "/srv/dev-disk-by-uuid-2c7cf803-58b4-430e-9c29-7cf25a1df1f4/home/aniruddha0516/DATA"


def main():
    ap = argparse.ArgumentParser(description="SSH into the data machine, inside the solar DATA folder")
    ap.add_argument("--user", default=USER)
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--dir", default=DATA)
    a = ap.parse_args()
    os.system(f"title SUIT data machine - {a.user}@{a.host}")
    ssh = shutil.which("ssh") or r"C:\Windows\System32\OpenSSH\ssh.exe"
    if not os.path.exists(ssh):
        print("The Windows OpenSSH client (ssh.exe) is not installed.")
        input("Press Enter to close...")
        return 1
    print(f"SUIT data machine  {a.user}@{a.host}\nfolder  {a.dir}\n")
    remote = (f"cd '{a.dir}' 2>/dev/null && echo 'Now in the solar DATA folder:' && pwd && ls "
              f"|| echo 'DATA folder not found: {a.dir}'; exec bash")
    rc = subprocess.call([ssh, "-t", "-o", "ServerAliveInterval=30", "-o", "ConnectTimeout=10",
                          f"{a.user}@{a.host}", remote])
    if rc not in (0, 130):
        print(f"\nssh ended with code {rc}.")
        if rc == 255:
            print("  - 'Permission denied': wrong user name or password (the user is 'aniruddha', not 'aniruddha0516')\n"
                  "  - 'Connection timed out/refused': the data machine is off, or SSH is off in OpenMediaVault\n"
                  "    (Services > SSH)")
        input("Press Enter to close...")
    return rc


if __name__ == "__main__":
    sys.exit(main())
