import gradio as gr
from document_processor import DocumentProcessor
from rag_engine import RAGEngine

doc_processor = None
rag_engine = None

def process_files(files):
    """Process uploaded documents"""
    global doc_processor, rag_engine

    if not files:
        return "No files uploaded"

    file_paths = [file.name for file in files]
    doc_processor = DocumentProcessor()
    doc_processor.process_documents(file_paths)

    rag_engine = RAGEngine(doc_processor.vectorstore)

    return f"Processed {len(file_paths)} documents successfully!"

def chat(message, history):
    """Handle chat messages"""
    if rag_engine is None:
        return "Please upload documents first!"

    result = rag_engine.query(message)
    return result["answer"]

with gr.Blocks() as demo:
    gr.Markdown("# RAG Document Chat")

    with gr.Tab("Upload Documents"):
        file_upload = gr.File(
            file_count="multiple",
            label="Upoload Documents (pdf or txt)"
        )
        upload_btn = gr.Button(label="Process Documents")
        upload_status = gr.Textbox(label="Status")

        upload_btn.click(
            fn=process_files,
            inputs=file_upload,
            outputs=upload_status
        )

        with gr.Tab("Chat"):
            chatbot = gr.ChatInterface(
                fn=chat,
                title="Ask questions about your documents",
                examples=[
                    "What is the main topic of the documents?",
                    "Summarize the key points",
                    "What does my document say about ...?"
                ]
            )

if __name__ == "__main__":
    demo.launch()