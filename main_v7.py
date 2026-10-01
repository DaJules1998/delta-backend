import os
import json
import tempfile
from datetime import datetime
import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from openai import OpenAI

# --- NEUE IMPORTS FÜR DB & AUTH ---
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from passlib.context import CryptContext
from jose import JWTError, jwt

# --- 1. SETUP & DATENBANK ---
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else OpenAI()

# Datenbank-Setup (SQLite für MVP)
SQLALCHEMY_DATABASE_URL = "sqlite:///./delta_checker.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Auth-Setup
SECRET_KEY = "dein-super-geheimer-mvp-schluessel-bitte-spaeter-aendern"
ALGORITHM = "HS256"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/login", auto_error=False)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. DATENBANK MODELLE ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    filename = Column(String)
    agent_type = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    result_json = Column(Text) # Speichert die extrahierten Items

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- 3. AUTHENTIFIZIERUNGS LOGIK ---
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
    except JWTError:
        return None
    return db.query(User).filter(User.username == username).first()

# --- 4. API-ROUTEN (ACCOUNT & HISTORIE) ---
@app.post("/api/register")
def register(user_data: dict, db: Session = Depends(get_db)):
    username = user_data.get("username")
    password = user_data.get("password")
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=400, detail="Benutzername bereits vergeben.")
    
    new_user = User(username=username, hashed_password=get_password_hash(password))
    db.add(new_user)
    db.commit()
    return {"message": "Account erfolgreich erstellt!"}

@app.post("/api/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Falscher Benutzername oder Passwort")
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/history")
def get_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Nicht eingeloggt")
    reports = db.query(Report).filter(Report.user_id == current_user.id).order_by(Report.created_at.desc()).all()
    
    history = []
    for r in reports:
        history.append({
            "id": r.id,
            "filename": r.filename,
            "type": r.agent_type,
            "date": r.created_at.strftime("%d.%m.%Y %H:%M"),
            "items": json.loads(r.result_json)
        })
    return {"history": history}

# --- 5. DAS KI-GEHIRN (VOLLSTÄNDIGER PROMPT) ---
SYSTEM_PROMPT = """Du bist ein österreichischer Bauingenieur (HTL Hochbau/Baumeister) und Prüfer.
Analysiere den Text und finde JEDE EINZELNE bautechnische Nachweispflicht. 

INHALTLICHER TÜRSTEHER (NOTFALL-ABBRUCH):
Wenn der übergebene Text absolut NICHTS mit dem Bauwesen, Ausschreibungen oder bautechnischen Bescheiden zu tun hat (z. B. ein Kochrezept, ein Roman, ein privater Brief), dann brich die Analyse für diesen Abschnitt sofort ab und antworte EXAKT mit diesem JSON:
{"items": [{"stelle": "Gesamtes Dokument", "dokument": "Abbruch: Fachfremd", "logik": "Der Text hat keinen bautechnischen Bezug.", "einstufung": "Prüfen"}]}

WICHTIGSTE REGELN ZU POSITIONSNUMMERN (z.B. nach LB-HB):
1. ÖSTERREICHISCHE POSITIONEN SIND 6-STELLIG: Eine exakte Position besteht aus 6 Ziffern und oft einem Buchstaben am Ende. Beispiele: "00.10.17", "03.12.04" oder "00.10.25.A".
2. ABSOLUTES BÜNDELUNGSVERBOT: Du darfst NIEMALS Dokumente auf Obergruppen (wie "Pos 02" oder "Pos 03") zusammenfassen! Du musst IMMER die exakte, 6-stellige Nummer (inkl. Buchstabe) aus dem Text suchen. 

DOKUMENTATIONS-REGELN:
- JEDES MATERIAL: Für JEDES gelieferte Material (egal ob Beton, Holz, Ziegel, Rohre, Türen etc.) MUSS ein allgemeingültiger Eintrag "Lieferschein / Leistungserklärung" erstellt werden.
- JEDE ENTSORGUNG: Für JEDEN Aushub/Abbruch MUSS ein allgemeingültiger "Wiegeschein / Entsorgungsnachweis" gefordert werden.
- DOKUMENTE SPLITTEN (SEHR WICHTIG): Braucht eine Position mehrere Dokumente (z.B. Lieferschein UND Einbaubestätigung), musst du für JEDES Dokument eine EIGENE, separate Zeile im JSON (mit derselben Positionsnummer) ausgeben! Fasse niemals mehrere Kategorien in einem Feld zusammen.

ANTWORTE ZWINGEND IM JSON-FORMAT!
Struktur: {"items": [{"stelle": "Exakte Pos-Nummer: Kurztext", "dokument": "Name des Dokuments", "logik": "Begründung", "einstufung": "Zwingend/Implizit/Potenziell"}]}

### BEISPIEL FÜR PERFEKTE EXTRAKTION AUS EINEM LV ###
USER: 00.10.25.A Sub-/Nachunternehmer zulässig ... 03.12.04 Brandschutztürelement EI2 30-C ...
ASSISTANT: {"items": [
  {"stelle": "Pos. 00.10.25.A: Sub-/Nachunternehmer", "dokument": "Nachweis der Befugnis und Leistungsfähigkeit", "logik": "Bieter muss Befugnisse der Subunternehmer nachweisen.", "einstufung": "Zwingend"},
  {"stelle": "Pos. 03.12.04: Brandschutztürelement EI2 30-C", "dokument": "Lieferschein / Leistungserklärung", "logik": "Jedes Material erfordert einen Materialnachweis.", "einstufung": "Implizit"},
  {"stelle": "Pos. 03.12.04: Brandschutztürelement EI2 30-C", "dokument": "Einbaubestätigung", "logik": "Brandschutznachweis zwingend erforderlich für Benützungsfreigabe.", "einstufung": "Zwingend"}
]}
"""

def chunk_text(text: str, chunk_size: int = 30000, overlap: int = 1000):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

# --- 6. DIE KERN-ANALYSE (Mit allen Türstehern & Speichern-Funktion) ---
@app.post("/api/analyze")
async def analyze_document(
    file: UploadFile = File(...), 
    agent_type: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    print(f"\n[+] Eingehende Analyseanfrage: {file.filename} von User: {current_user.username if current_user else 'Gast'}")
    
    # Türsteher 1: Dateiendung
    if not file.filename.lower().endswith('.pdf'):
        return {"items": [{"stelle": "Sicherheits-Abbruch", "dokument": "Falsches Format", "logik": "Nur PDF erlaubt.", "einstufung": "Prüfen"}]}

    tmp_file_path = None
    MAX_FILE_SIZE = 15 * 1024 * 1024 
    
    try:
        # Türsteher 2: Dateigröße (15MB Limit) und RAM-schonend auf Festplatte streamen
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file_path = tmp_file.name
            content_length = 0
            while chunk := await file.read(1024 * 1024):
                content_length += len(chunk)
                if content_length > MAX_FILE_SIZE:
                    os.remove(tmp_file_path)
                    return {"items": [{"stelle": "Abbruch", "dokument": "Zu groß", "logik": "Datei überschreitet 15 MB.", "einstufung": "Prüfen"}]}
                tmp_file.write(chunk)
                
        # Türsteher 3: Echter PDF-Header (Blockiert umbenannte Videos/Bilder)
        with open(tmp_file_path, 'rb') as f:
            if f.read(5) != b'%PDF-':
                os.remove(tmp_file_path)
                return {"items": [{"stelle": "Abbruch", "dokument": "Fake-PDF", "logik": "Die Datei ist kein echtes PDF.", "einstufung": "Prüfen"}]}
                
        # Text aus PDF extrahieren
        doc = fitz.open(tmp_file_path)
        pdf_text = "\n".join([page.get_text() for page in doc])
        doc.close()
        os.remove(tmp_file_path)

    except Exception as e:
        if tmp_file_path and os.path.exists(tmp_file_path): os.remove(tmp_file_path)
        return {"items": [{"stelle": "Abbruch", "dokument": "Dateifehler", "logik": str(e), "einstufung": "Prüfen"}]}
    
    if len(pdf_text.strip()) < 20:
        return {"items": [{"stelle": "Abbruch", "dokument": "Kein Text", "logik": "Dokument besteht nur aus Bildern/Scans.", "einstufung": "Prüfen"}]}
    
    chunks = chunk_text(pdf_text)
    all_items = []
    
    # KI Analyse
    for i, chunk in enumerate(chunks):
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
            extracted = json.loads(response.choices[0].message.content).get("items", [])
            all_items.extend(extracted)
        except Exception as e:
            print(f"[-] Fehler in Block {i+1}: {e}")
            
    # Türsteher 4 (Inhaltlich): Fachfremd-Erkennung auswerten
    fachfremd = [i for i in all_items if "Fachfremd" in str(i.get("dokument", ""))]
    if len(fachfremd) > 0 and len(fachfremd) == len(all_items):
        return {"items": [{"stelle": "Inhalts-Abbruch", "dokument": "Fachfremd", "logik": "Kein bautechnischer Bezug.", "einstufung": "Prüfen"}]}
    
    all_items = [i for i in all_items if "Fachfremd" not in str(i.get("dokument", ""))]

    # Datenbank-Speicherung für angemeldete Nutzer
    if current_user and len(all_items) > 0:
        new_report = Report(
            user_id=current_user.id,
            filename=file.filename,
            agent_type=agent_type,
            result_json=json.dumps(all_items)
        )
        db.add(new_report)
        db.commit()
        print(f"[+] Bericht in Datenbank für '{current_user.username}' gespeichert.")

    return {"items": all_items}

@app.get("/")
async def serve_frontend():
    if os.path.exists("delta-checker_v9.html"):
        return FileResponse("delta-checker_v9.html")
    return {"message": "API läuft. Frontend wird über Netlify gehostet."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
