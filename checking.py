import pkg_resources
# 獲取所有已安裝的套件
installed_packages = pkg_resources.working_set
# 將套件名稱和版本格式化為 requirements.txt 格式
requirements = [f"{package.project_name}=={package.version}" for package in installed_packages]
# 將結果寫入 requirements.txt 檔案
with open("requirements.txt", "w") as f:
    for requirement in requirements:
        f.write(requirement + "\n")
print("requirements.txt 已生成，包含所有已安裝的套件及其版本。")