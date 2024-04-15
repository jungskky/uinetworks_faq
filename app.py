
import gradio as gr
from util import querying

iface = gr.ChatInterface(
    fn = querying,
    chatbot=gr.Chatbot(height=600),
    textbox=gr.Textbox(placeholder="질문을 입력해 주세요.", container=False, scale=7),
    title="쿠팡FAQ [(주)유아이네트웍스 AI 챗봇]",
    theme="soft",
    examples=["유효기간 안에 사용하지 못하면 환불 받을 수 있을까요?",
              "무통장입금을 하고 주문을 취소했는데 환불은 언제 되나요?",],

    cache_examples=True,
    retry_btn="Retry",
    undo_btn="Undo",
    clear_btn="Clear",
    submit_btn="Submit"

    )

iface.launch(share=True)

def on_close():
  iface.set_on_close(on_close)
  iface.launch()
  iface.close()