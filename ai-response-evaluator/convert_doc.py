from markdown_pdf import MarkdownPdf
from markdown_pdf import Section

def convert_md_to_pdf(input_path, output_path):
    pdf = MarkdownPdf(toc_level=2)
    with open(input_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
        
    pdf.add_section(Section(md_content))
    pdf.save(output_path)
    print(f"PDF generated successfully at: {output_path}")

if __name__ == "__main__":
    convert_md_to_pdf(
        "docs/Technical_Documentation.md", 
        "docs/Technical_Documentation.pdf"
    )
