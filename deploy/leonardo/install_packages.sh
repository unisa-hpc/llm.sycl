# Install OneAPI with Online Installer
cd /tmp
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/9a98af19-1c68-46ce-9fdd-e249240c7c42/l_BaseKit_p_2024.2.0.634.sh
sh ./l_BaseKit_p_2024.2.0.634.sh -a --silent --cli --eula accept

# Install CodePlay Nvidia Plugin
curl -LOJ "https://developer.codeplay.com/api/v1/products/download?product=oneapi&variant=nvidia&version=2024.2.0&filters[]=12.0&filters[]=linux"

# Source OneAPI then run the installer
source ~/intel/oneapi/setvars.sh
bash oneapi-for-nvidia-gpus-2024.2.0-cuda-12.0-linux.sh

