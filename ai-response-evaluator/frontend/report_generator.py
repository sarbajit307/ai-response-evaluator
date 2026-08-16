from fpdf import FPDF
import pandas as pd
from datetime import datetime
import json

class PDFReport(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 15)
        self.cell(0, 10, "AI Response Evaluation Report", border=0, align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("helvetica", "I", 10)
        self.cell(0, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", border=0, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

def generate_pdf_report(df: pd.DataFrame, report_title="Batch Evaluation Summary") -> bytes:
    pdf = PDFReport()
    pdf.add_page()
    
    if df.empty:
        pdf.set_font("helvetica", size=12)
        pdf.cell(0, 10, "No evaluation data available.", new_x="LMARGIN", new_y="NEXT")
        return pdf.output()

    pdf.set_font("helvetica", size=12)

    # 1. Project Details & Batch Summary
    total_evals = len(df)
    pass_count = len(df[df['final_verdict'] == 'Pass'])
    needs_imp_count = len(df[df['final_verdict'] == 'Needs Improvement'])
    fail_count = len(df[df['final_verdict'] == 'Fail'])

    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, report_title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", size=11)
    
    pdf.cell(0, 8, f"Total Evaluations: {total_evals}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Pass: {pass_count} | Needs Improvement: {needs_imp_count} | Fail: {fail_count}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # Averages
    if "overall_score" in df.columns:
        avg_score = df["overall_score"].mean()
        pdf.cell(0, 8, f"Average Overall Score: {avg_score:.2f} / 10.0", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)

    # 2. Individual Evaluations
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "Individual Evaluation Breakdowns", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    for idx, row in df.iterrows():
        pdf.set_font("helvetica", "B", 12)
        eval_id = row.get('id', idx+1)
        verdict = row.get('final_verdict', 'N/A')
        score = row.get('overall_score', 0.0)
        pdf.cell(0, 8, f"Eval ID {eval_id}: {verdict} (Score: {score})", new_x="LMARGIN", new_y="NEXT")
        
        pdf.set_font("helvetica", "I", 10)
        q_text = str(row.get('question', ''))[:300] + "..." if len(str(row.get('question', ''))) > 300 else str(row.get('question', ''))
        r_text = str(row.get('response', ''))[:300] + "..." if len(str(row.get('response', ''))) > 300 else str(row.get('response', ''))
        pdf.multi_cell(0, 6, f"Q: {q_text}", new_x="LMARGIN", new_y="NEXT")
        pdf.multi_cell(0, 6, f"A: {r_text}", new_x="LMARGIN", new_y="NEXT")
        
        pdf.set_font("helvetica", size=10)
        pdf.multi_cell(0, 6, f"Reasoning: {row.get('synthesis', '')}", new_x="LMARGIN", new_y="NEXT")
        
        # Hallucinations
        hallucinations = row.get("hallucinated_statements", [])
        if isinstance(hallucinations, str):
            try:
                hallucinations = json.loads(hallucinations)
            except:
                hallucinations = []
                
        if hallucinations:
            pdf.set_font("helvetica", "B", 10)
            pdf.cell(0, 6, "Flagged Hallucinations:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("helvetica", size=10)
            for stmt in hallucinations:
                pdf.multi_cell(0, 6, f"- {stmt}", new_x="LMARGIN", new_y="NEXT")
        
        pdf.ln(5)

    return pdf.output()
