import sys
import argparse

print("hi")
print(sys.version)

parser = argparse.ArgumentParser(description ="사용법 테스트")
parser.add_argument("--env_path" , type = str , help ="타겟값")
parser.add_argument("--script_folder" , type = str , help ="input")
parser.add_argument("--env" , type = str , help ="input")
parser.add_argument("--target" , type = str , help ="타겟값")
parser.add_argument("--input" , type = str , help ="input")
arg = parser.parse_args()




print("codna 경로 : " , arg.env_path)
print("script 폴더 경로 : " , arg.script_folder)
print("가상 환경 이름 : " , arg.env)
print("인풋 : " , arg.input)
print("타겟 : " , arg.target)




