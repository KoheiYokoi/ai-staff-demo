import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. AIロジックの準備 (起動時に一度だけ実行) ---
if not os.getenv("OPENAI_API_KEY"):
    print("注意: OPENAI_API_KEYが環境変数に設定されていません。")

# 学習データ (デモ用：歯科医院マニュアル)
manual_text = """
【プロジェクト管理ツール "TaMate" ヘルプセンター】

1. 料金プランについて
   - Freeプラン: ユーザー数5名まで無料。ストレージ1GB。
   - Proプラン: 月額980円/ユーザー。プロジェクト数無制限。
   - Businessプラン: 月額1,980円/ユーザー。SSO連携、監査ログ機能あり。

2. タスクの共有方法
   - タスク詳細画面の右上にある「共有ボタン」をクリックし、招待リンクをコピーしてください。
   - 外部ゲスト（クライアント等）も閲覧のみなら無料で招待可能です。

3. よくあるトラブル
   - ログインできない場合: パスワードリセットメールが迷惑メールフォルダに入っていないか確認してください。
   - 通知が届かない場合: プロフィール設定の「通知設定」でSlack連携がONになっているか確認してください。
   - API連携: Proプラン以上でAPIキーを発行可能です。設定画面 > API から発行できます。
"""

# ベクトル化
docs = [Document(page_content=manual_text)]
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(docs, embeddings)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# プロンプト設定
prompt = ChatPromptTemplate.from_template("""
あなたはプロジェクト管理ツール「TaskMate」のカスタマーサポートです。
以下の【マニュアル】に基づいて、ユーザーの質問に回答してください。
マニュアルにない技術的な質問は「サポート担当（support@taskmate.com）へお問い合わせください」と案内してください。

【マニュアル】
{context}

質問: {input}
""")

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vector_store.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# --- 2. FastAPIサーバーの設定 ---
app = FastAPI()

# CORS設定 (HTMLファイルからAPIを叩けるように許可する設定)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では特定のドメインのみ許可すべき
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# リクエストボディの定義
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        # AIに質問を投げる
        response = retrieval_chain.invoke({"input": req.message})
        return {"reply": response['answer']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # localhost:8000 でサーバーを起動
    uvicorn.run(app, host="0.0.0.0", port=8000)