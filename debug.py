import sys
import os
# from eval import main as main_eval
import exec
sys.argv = ["exec.py", "--disable_cntx_agenda","--disable_value_pred","--load_less","--gpu", "1"]
# sys.argv = ["exec.py", "--disable_cntx_agenda","--gpu", "2"]
sys.argv = ["exec.py", "--disable_cntx_agenda","--is_oracle","--gpu", "2"]
# sys.argv += ["--debug"]
print(sys.argv)
print(" ".join(sys.argv))
exec.run()
