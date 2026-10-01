import os
import json
import fitz  # PyMuPDF für schnelle PDF-Textextraktion
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from openai import OpenAI

# --- 1. SETUP CLOUD-READY ---
# Holt den Key automatisch aus den Render-Umgebungsvariablen (.env)
# Hier steht KEIN Key mehr im Klartext!
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else OpenAI()

app = FastAPI()

# Extrem wichtig für Netlify (erlaubt Anfragen von einer anderen Domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. DAS GEHIRN (Dein Few-Shot Spickzettel) ---
SYSTEM_PROMPT = """Du bist ein österreichischer Bauingenieur (HTL Hochbau/Baumeister) und Prüfer.
Analysiere den Text und finde JEDE EINZELNE bautechnische Nachweispflicht. 

WICHTIGSTE REGELN ZU POSITIONSNUMMERN (z.B. nach LB-HB):
1. ÖSTERREICHISCHE POSITIONEN SIND 6-STELLIG: Eine exakte Position besteht aus 6 Ziffern und oft einem Buchstaben am Ende. Beispiele: "00.10.17", "03.12.04" oder "00.10.25.A".
2. ABSOLUTES BÜNDELUNGSVERBOT: Du darfst NIEMALS Dokumente auf Obergruppen (wie "Pos 02" oder "Pos 03") zusammenfassen! Du musst IMMER die exakte, 6-stellige Nummer (inkl. Buchstabe) aus dem Text suchen. 

DOKUMENTATIONS-REGELN:
- JEDES MATERIAL: Für JEDES gelieferte Material (egal ob Beton, Holz, Ziegel, Rohre, Türen etc.) MUSS ein Eintrag "Lieferschein / Leistungserklärung" erstellt werden.
- JEDE ENTSORGUNG: Für JEDEN Aushub/Abbruch MUSS ein "Wiegeschein / Entsorgungsnachweis" gefordert werden.
- DOKUMENTE SPLITTEN: Braucht eine Position Lieferschein UND Zertifikat, mache ZWEI (oder mehr) separate Einträge mit derselben Positionsnummer!

ANTWORTE ZWINGEND IM JSON-FORMAT!
Struktur: {"items": [{"stelle": "Exakte Pos-Nummer: Kurztext", "dokument": "Name des Dokuments", "logik": "Begründung", "einstufung": "Zwingend/Implizit/Potenziell"}]}

### BEISPIEL FÜR PERFEKTE EXTRAKTION AUS EINEM LV ###
USER: 00.10.25.A Sub-/Nachunternehmer zulässig ... 03.12.04 Brandschutztürelement EI2 30-C ...
ASSISTANT: {"items": [
  {"stelle": "Pos. 00.10.25.A: Sub-/Nachunternehmer zulässig", "dokument": "Nachweis der Befugnis und Leistungsfähigkeit", "logik": "Bieter muss Befugnisse der Subunternehmer nachweisen.", "einstufung": "Zwingend"},
  {"stelle": "Pos. 03.12.04: Brandschutztürelement EI2 30-C", "dokument": "Lieferschein / Leistungserklärung", "logik": "Jedes Material erfordert einen Nachweis.", "einstufung": "Implizit"},
  {"stelle": "Pos. 03.12.04: Brandschutztürelement EI2 30-C", "dokument": "Einbaubestätigung", "logik": "Brandschutznachweis zwingend erforderlich für Benützungsfreigabe.", "einstufung": "Zwingend"}
]}
"""

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extrahiert den Text aus allen Seiten des PDFs."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = "\n".join([page.get_text() for page in doc])
    return text

# CHUNKING: Zerschneidet lange Dokumente in blöcke von ca. 10 Seiten
def chunk_text(text: str, chunk_size: int = 30000, overlap: int = 1000) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# --- 3. DIE API-SCHNITTSTELLE ---
@app.post("/api/analyze")
async def analyze_document(file: UploadFile = File(...), agent_type: str = Form(...)):
    print(f"\n[+] Eingehende Analyseanfrage: {file.filename} (Typ: {agent_type})")
    
    try:
        content = await file.read()
        pdf_text = extract_text_from_pdf(content)
    except Exception as e:
        return JSONResponse(status_code=400, content={"items": [{"stelle": "Fehler beim PDF-Lesen", "dokument": "Abbruch", "logik": str(e), "einstufung": "Prüfen"}]})
    
    if len(pdf_text.strip()) < 20:
        return {"items": [{"stelle": "Gesamtes Dokument", "dokument": "Kein Text", "logik": "Das PDF besteht nur aus Fotos/Scans.", "einstufung": "Prüfen"}]}
    
    # Text in verdauliche Blöcke zerteilen
    chunks = chunk_text(pdf_text)
    print(f"[*] Dokument in {len(chunks)} Abschnitte zerschnitten. Starte iterative Analyse...")
    
    all_items = []
    
    # Die Schleife jagt jeden Block einzeln durch die KI
    for i, chunk in enumerate(chunks):
        print(f"    -> Analysiere Block {i+1}/{len(chunks)}...")
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                temperature=0.1, 
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Analysiere diesen Textabschnitt:\n\n{chunk}"}
                ]
            )
            result = json.loads(response.choices[0].message.content)
            extracted = result.get("items", [])
            all_items.extend(extracted)
            print(f"       Erfolg: {len(extracted)} Einträge gefunden.")
        except Exception as e:
            print(f"       [-] Fehler in Block {i+1}: {e}")
            
    print(f"[+] Gesamtanalyse fertig! {len(all_items)} Einträge an das Frontend übermittelt.")
    return {"items": all_items}

# --- 4. FALLBACK ROUTE ---
@app.get("/")
async def serve_frontend():
    # Wenn die Datei lokal liegt, zeige sie an. In der Cloud (Render) wird hier nur gemeldet, dass die API läuft.
    if os.path.exists("delta_checker_v7.html"):
        return FileResponse("delta_checker_v7.html")
    return {"message": "Delta-Checker Backend läuft. Frontend wird über Netlify gehostet."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)