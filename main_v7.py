import os
import json
import tempfile
import fitz  # PyMuPDF für schnelle PDF-Textextraktion
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from openai import OpenAI

# --- 1. SETUP CLOUD-READY ---
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else OpenAI()

app = FastAPI()

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

INHALTLICHER TÜRSTEHER (NOTFALL-ABBRUCH):
Wenn der übergebene Text absolut NICHTS mit dem Bauwesen, Ausschreibungen oder bautechnischen Bescheiden zu tun hat (z. B. ein Kochrezept, ein Roman, ein privater Brief), dann brich die Analyse für diesen Abschnitt sofort ab und antworte EXAKT mit diesem JSON:
{"items": [{"stelle": "Gesamtes Dokument", "dokument": "Abbruch: Fachfremd", "logik": "Der Text hat keinen bautechnischen Bezug.", "einstufung": "Prüfen"}]}

WICHTIGSTE REGELN ZU POSITIONSNUMMERN (z.B. nach LB-HB):
1. ÖSTERREICHISCHE POSITIONEN SIND 6-STELLIG: Eine exakte Position besteht aus 6 Ziffern und oft einem Buchstaben am Ende. Beispiele: "00.10.17", "03.12.04" oder "00.10.25.A".
2. ABSOLUTES BÜNDELUNGSVERBOT: Du darfst NIEMALS Dokumente auf Obergruppen zusammenfassen! Du musst IMMER die exakte, 6-stellige Nummer aus dem Text suchen. 

DOKUMENTATIONS-REGELN:
- JEDES MATERIAL: Für JEDES gelieferte Material (egal ob Beton, Holz, Ziegel, Rohre, Türen etc.) MUSS ein Eintrag "Lieferschein / Leistungserklärung" erstellt werden.
- JEDE ENTSORGUNG: Für JEDEN Aushub/Abbruch MUSS ein "Wiegeschein / Entsorgungsnachweis" gefordert werden.
- DOKUMENTE SPLITTEN: Braucht eine Position Lieferschein UND Zertifikat, mache ZWEI (oder mehr) separate Einträge!

ANTWORTE ZWINGEND IM JSON-FORMAT!
Struktur: {"items": [{"stelle": "Exakte Pos-Nummer: Kurztext", "dokument": "Name des Dokuments", "logik": "Begründung", "einstufung": "Zwingend/Implizit/Potenziell"}]}
"""

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
    
    # TÜRSTEHER 1: Dateiendung
    if not file.filename.lower().endswith('.pdf'):
        print("[-] Blockiert: Falsche Endung.")
        return {"items": [{"stelle": "Sicherheits-Abbruch", "dokument": "Falsches Format", "logik": "Es dürfen nur PDF-Dateien hochgeladen werden.", "einstufung": "Prüfen"}]}

    tmp_file_path = None
    MAX_FILE_SIZE = 15 * 1024 * 1024  # 15 MB Limit (Schützt den RAM)
    
    try:
        # TÜRSTEHER 2: RAM-schonendes Streaming auf die Festplatte
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file_path = tmp_file.name
            content_length = 0
            while chunk := await file.read(1024 * 1024):
                content_length += len(chunk)
                if content_length > MAX_FILE_SIZE:
                    os.remove(tmp_file_path)
                    print("[-] Blockiert: Datei zu groß.")
                    return {"items": [{"stelle": "Sicherheits-Abbruch", "dokument": "Datei zu groß", "logik": "Die Datei überschreitet das Limit von 15 MB.", "einstufung": "Prüfen"}]}
                tmp_file.write(chunk)
                
        # TÜRSTEHER 3: Ist es wirklich ein echtes PDF? (Checkt den Header-Code)
        with open(tmp_file_path, 'rb') as f:
            header = f.read(5)
            if header != b'%PDF-':
                os.remove(tmp_file_path)
                print("[-] Blockiert: Datei ist kein echtes PDF (Fake).")
                return {"items": [{"stelle": "Sicherheits-Abbruch", "dokument": "Fake-PDF", "logik": "Die Datei ist kein echtes PDF, sondern wurde manipuliert.", "einstufung": "Prüfen"}]}
                
        # Text extrahieren
        doc = fitz.open(tmp_file_path)
        pdf_text = "\n".join([page.get_text() for page in doc])
        doc.close()
        
        # Festplatte wieder aufräumen
        os.remove(tmp_file_path)

    except Exception as e:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
        print(f"[-] Blockiert: Dateifehler ({e})")
        return {"items": [{"stelle": "Sicherheits-Abbruch", "dokument": "Datei defekt", "logik": f"Fehler: {str(e)}", "einstufung": "Prüfen"}]}
    
    if len(pdf_text.strip()) < 20:
        return {"items": [{"stelle": "Gesamtes Dokument", "dokument": "Kein Text", "logik": "Das PDF besteht nur aus Scans ohne lesbaren Text.", "einstufung": "Prüfen"}]}
    
    chunks = chunk_text(pdf_text)
    print(f"[*] Dokument in {len(chunks)} Abschnitte zerschnitten. Starte iterative Analyse...")
    
    all_items = []
    
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
            
    # --- INTELLIGENTE AUSWERTUNG DES INHALTLICHEN TÜRSTEHERS ---
    fachfremd_eintraege = [item for item in all_items if "Fachfremd" in str(item.get("dokument", ""))]
    
    # Abbruch, wenn das Dokument komplett fachfremd ist
    if len(fachfremd_eintraege) > 0 and len(fachfremd_eintraege) == len(all_items):
        print("[-] Blockiert: KI hat das Dokument als völlig fachfremd eingestuft.")
        return {"items": [{"stelle": "Inhalts-Abbruch", "dokument": "Fachfremdes Dokument", "logik": "Die KI hat erkannt, dass dieses Dokument keinen bautechnischen Bezug hat (z. B. Kochrezept, Roman).", "einstufung": "Prüfen"}]}
    
    # Ansonsten ignorieren wir einzelne fachfremde Chunks (z.B. Impressum) und leiten die echten Ergebnisse weiter
    all_items = [item for item in all_items if "Fachfremd" not in str(item.get("dokument", ""))]

    print(f"[+] Gesamtanalyse fertig! {len(all_items)} echte bautechnische Einträge übermittelt.")
    return {"items": all_items}

# --- 4. FALLBACK ROUTE ---
@app.get("/")
async def serve_frontend():
    if os.path.exists("delta-checker_v8.html"):
        return FileResponse("delta-checker_v8.html")
    return {"message": "Delta-Checker Backend läuft. Frontend wird über Netlify gehostet."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
