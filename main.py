import langchain
import langchain_core

def main():
    print("=" * 40)
    print("LangChain 설치 확인")
    print("=" * 40)
    print(f"LangChain 버전      : {langchain.__version__}")
    print(f"LangChain Core 버전 : {langchain_core.__version__}")
    print("=" * 40)
    print("정상적으로 설치되어 로드되었습니다!")

if __name__ == "__main__":
    main()

