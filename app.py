import re
import joblib
import gradio as gr
from scipy.sparse import hstack

def clean_arabic_text(text):
    text = str(text)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002700-\U000027BF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA70-\U0001FAFF"
        "\U00002600-\U000026FF"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(" ", text)
    punct_pattern = r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~،؛؟…“”‘’••]'
    text = re.sub(punct_pattern, " ", text)
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text, flags=re.UNICODE)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

model_cat = joblib.load("model_category_svm.joblib")
model_urg = joblib.load("model_urgency_logistic.joblib")
vec_w = joblib.load("tfidf_word.joblib")
vec_c = joblib.load("tfidf_char.joblib")

def predict_ui(text_input):
    lines = [line.strip() for line in text_input.splitlines() if line.strip()]
    if not lines:
        return "الرجاء كتابة شكوتك أولاً."

    output_blocks = []
    for idx, complaint in enumerate(lines, 1):
        cleaned = clean_arabic_text(complaint)
        vec = hstack([vec_w.transform([cleaned]), vec_c.transform([cleaned])])
        cat_pred = model_cat.predict(vec)[0]
        urg_pred = model_urg.predict(vec)[0]
        output_blocks.append(
            f"الشكوى {idx}: {complaint}\n"
            f"النوع: {cat_pred}\n"
            f"الأولوية: {urg_pred}"
        )

    return ("\n" + "-" * 40 + "\n").join(output_blocks)

with gr.Blocks() as demo:
    gr.Markdown("## تصنيف شكاوي العملاء")
    complaint_box = gr.Textbox(
        label="ماهي شكوتك",
        placeholder="اكتب شكوتك هنا...",
        lines=4
    )
    ok_btn = gr.Button("اوكي")
    result_box = gr.Textbox(
        label="النتيجة",
        lines=8,
        interactive=False
    )

    ok_btn.click(fn=predict_ui, inputs=complaint_box, outputs=result_box)

if __name__ == "__main__":
    demo.launch()

