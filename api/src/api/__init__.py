import unicodedata

from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


app = FastAPI(title="SPLADE Japanese API")

MODEL_ID = "aken12/splade-japanese-v3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading model {MODEL_ID} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForMaskedLM.from_pretrained(MODEL_ID).to(device)
model.eval()

# Elasticsearchで扱いやすいよう、IDから文字列トークンに戻すための辞書
id_to_token = {v: k for k, v in tokenizer.get_vocab().items()}
special_tokens = set(tokenizer.all_special_tokens)

# 検索にほとんど寄与しない助詞・助動詞などの日本語ストップワード
STOPWORDS = {
    "は", "が", "を", "に", "へ", "と", "で", "の", "や", "も", "か", "ね", "よ",
    "さ", "し", "て", "た", "だ", "な", "ら", "れ", "る", "ず", "ぬ",
    "から", "まで", "など", "および", "または", "ため", "こと", "もの",
    "これ", "それ", "あれ", "この", "その", "あの", "どの",
    "です", "ます", "である", "だっ", "でし", "まし",
    "する", "いる", "ある", "なる", "れる", "られる", "せる",
}


def _normalize_token(token: str) -> str:
    """サブワード接頭辞(##)を取り除いた中身を返す。"""
    return token[2:] if token.startswith("##") else token


def _is_stopword(token: str) -> bool:
    """検索ノイズになる記号・助詞などのトークンかどうかを判定する。"""
    core = _normalize_token(token)
    if not core:
        return True

    # 記号・句読点のみで構成されるトークンを除外（例: 。 、 ・ ( ) : ）
    if all(
        unicodedata.category(ch)[0] in ("P", "S", "Z", "C") for ch in core
    ):
        return True

    # ひらがな1文字（助詞など）を除外
    if len(core) == 1 and "\u3040" <= core <= "\u309f":
        return True

    # 明示的なストップワード
    if core in STOPWORDS:
        return True

    return False


class QueryRequest(BaseModel):
    text: str


@app.post("/encode")
def encode(req: QueryRequest, filterStopWords: bool = True):
    # テキストをトークナイズ
    inputs = tokenizer(req.text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    # SPLADE特有のプーリング処理: max(log(1 + ReLU(logits)) * attention_mask)
    attention_mask = inputs["attention_mask"].unsqueeze(-1)
    sparse_vec = torch.max(
        torch.log(1 + torch.relu(logits)) * attention_mask, dim=1
    ).values.squeeze()

    # 重みが0より大きい（非ゼロ）のインデックスを抽出
    non_zero_indices = sparse_vec.nonzero().squeeze(-1).tolist()
    if isinstance(non_zero_indices, int):
        non_zero_indices = [non_zero_indices]

    # Elasticsearch用の辞書を作成
    result_vector = {}
    for idx in non_zero_indices:
        weight = sparse_vec[idx].item()
        token = id_to_token[idx]

        # [CLS]や[PAD]などの特殊トークンは常に除外
        if token in special_tokens or weight <= 0.0:
            continue

        # filterStopWords=true のときのみ記号・助詞などのストップワードを除外
        if filterStopWords and _is_stopword(token):
            continue

        # 少数第4位で丸める（データ容量削減のため）
        result_vector[token] = round(weight, 4)

    # 見やすさのためweightの降順でソート（検索スコアには影響しない）
    sorted_vector = dict(
        sorted(result_vector.items(), key=lambda item: item[1], reverse=True)
    )

    return {"sparse_vector": sorted_vector}
