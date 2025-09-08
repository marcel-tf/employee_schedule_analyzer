from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
import os
import pandas as pd
import re
import mailbox
import html as ihtml
import unicodedata
from datetime import datetime
from werkzeug.utils import secure_filename
from email.header import decode_header
from email.utils import parseaddr

app = Flask(__name__, static_folder='static')
app.secret_key = 'your-secret-key-here'

# Configuration for uploaded files
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'mbox'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ==================== GOOGLE COLAB FUNCTIONS ====================

def normalize_name(name: str) -> str:
    """Normalize names by removing accents and converting to uppercase"""
    if not name:
        return ""
    # Convert to uppercase
    name = name.upper().strip()
    # Remove accents and special characters
    name = ''.join(
        c for c in unicodedata.normalize('NFD', name)
        if unicodedata.category(c) != 'Mn'
    )
    return name

def convert_csv_date_to_mbox_format(csv_date):
    """Convert CSV date to MBOX format"""
    try:
        dt = datetime.strptime(str(csv_date).strip(), '%a %b %d %Y')
        return dt.strftime('%Y-%m-%d')
    except:
        return str(csv_date).strip()

def flexible_name_match(csv_name, mbox_name):
    """
    Flexible matching rules:
    1. Exact match → True
    2. If not exact: Take first two words from MBOX
       and verify they are the first two from CSV
    """
    csv_name_norm = normalize_name(csv_name)
    mbox_name_norm = normalize_name(mbox_name)

    # Exact match
    if csv_name_norm == mbox_name_norm:
        return True

    # Controlled partial match
    csv_tokens = csv_name_norm.split()
    mbox_tokens = mbox_name_norm.split()

    if len(mbox_tokens) >= 2 and len(csv_tokens) >= 2:
        return csv_tokens[:2] == mbox_tokens[:2]

    return False

def exact_match(csv_name, csv_date, mbox_name, mbox_date):
    """Check if there's an exact match between CSV and MBOX"""
    if not csv_name or not mbox_name:
        return False

    # Flexible name matching
    name_match = flexible_name_match(csv_name, mbox_name)

    # Date matching
    csv_date_converted = convert_csv_date_to_mbox_format(csv_date)
    date_match = csv_date_converted == str(mbox_date).strip()

    return name_match and date_match

def process_csv_file(csv_path):
    """Process CSV file"""
    try:
        df = pd.read_csv(csv_path, index_col=0)
        df = df.reset_index().rename(columns={'index': 'Employee'})
        df.columns = [c.strip() for c in df.columns]

        # Rename specific columns by position
        column_names = df.columns.tolist()
        if len(column_names) >= 5:
            df.columns.values[1] = 'hour_type'
            df.columns.values[2] = 'daydate'
            df.columns.values[3] = 'In'
            df.columns.values[4] = 'out'

        # Capture unpaid break (8th column) if exists
        unpaid_break_col = None
        if len(df.columns) >= 8:
            unpaid_break_col = df.columns[7]  # 8th column (index 7)

        # Select columns we need
        if unpaid_break_col:
            df_selected = df[['Employee', 'hour_type', 'daydate', 'In', 'out', unpaid_break_col]].copy()
            df_selected.rename(columns={unpaid_break_col: 'unpaid_break'}, inplace=True)
        else:
            df_selected = df.iloc[:, :5].copy()
            df_selected['unpaid_break'] = ''

        # Normalize names
        df_selected['Employee'] = df_selected['Employee'].apply(normalize_name)

        # Format In and out columns
        def format_time(time_str):
            if pd.isna(time_str) or time_str == '':
                return ''
            return str(time_str).strip()

        df_selected['In'] = df_selected['In'].apply(format_time)
        df_selected['out'] = df_selected['out'].apply(format_time)
        df_selected['unpaid_break'] = df_selected['unpaid_break'].apply(format_time)

        print(f"✅ CSV processed: {len(df_selected)} rows")
        return df_selected

    except Exception as e:
        print(f"❌ Error processing CSV: {e}")
        return None

# Functions for processing MBOX (from Google Colab)
LOGIN_PHRASE = "I will be logging in for today"
LOGOUT_PHRASE = "I will be logging out for today"
LUNCH_START_PHRASE = "Hi team! I will be taking my lunch break."
LUNCH_END_PHRASE = "Hi team! I am back from my lunch break."

SENT_RE = re.compile(
    r'([A-Za-z]+)\s+(\d{1,2}),\s+(\d{4})\s+at\s+' 
    r'(\d{1,2}:\d{2}(?::\d{2})?)\s*([AP]M)' 
    r'(?:\s*GMT[^\s]*)?',
    re.I
)

NAME_RE = re.compile(r'Name:\s*([A-Za-zÀÁÉÍÓÚÜÇÑ\-\.\']+(?:\s+[A-Za-zÀÁÉÍÓÚÜÇÑ\-\.\']+)*)', re.I)

def decode_header_value(val: str) -> str:
    """Decode email headers"""
    if not val:
        return ""
    parts = decode_header(val)
    out = []
    for bytes_or_str, charset in parts:
        if isinstance(bytes_or_str, bytes):
            cs = charset or "utf-8"
            try:
                out.append(bytes_or_str.decode(cs, errors="replace"))
            except:
                out.append(bytes_or_str.decode("utf-8", errors="replace"))
        else:
            out.append(bytes_or_str)
    return "".join(out)

def get_body_text(msg) -> str:
    """Extract text from message body"""
    candidates = []
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_maintype() == 'multipart':
                continue
            disp = part.get("Content-Disposition", "") or ""
            if disp.strip().lower().startswith("attachment"):
                continue
            ctype = (part.get_content_type() or "").lower()
            if ctype == "text/plain":
                candidates.append(("plain", part))
            elif ctype == "text/html":
                candidates.append(("html", part))
    else:
        ctype = (msg.get_content_type() or "").lower()
        if ctype in ("text/plain", "text/html"):
            candidates.append(("plain" if ctype=="text/plain" else "html", msg))

    candidates.sort(key=lambda t: 0 if t[0]=="plain" else 1)

    for kind, part in candidates:
        try:
            payload = part.get_payload(decode=True)
            if payload is None:
                continue
            charset = part.get_content_charset() or "utf-8"
            text = payload.decode(charset, errors="replace")
            if kind == "html":
                text = re.sub(r'<script\b[^>]*>.*?</script>', ' ', text, flags=re.I|re.S)
                text = re.sub(r'<style\b[^>]*>.*?</style>', ' ', text, flags=re.I|re.S)
                text = re.sub(r'<[^>]+>', ' ', text)
                text = ihtml.unescape(text)
            text = text.replace('\u202f',' ').replace('\u200b',' ').replace('\xa0',' ')
            text = re.sub(r'=\r?\n', '', text)
            text = re.sub(r'=\s+', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            if text:
                return text
        except Exception:
            continue
    return ""

def clean_name(raw_name: str) -> str:
    """Clean name to keep only first two names"""
    if not raw_name:
        return ""
    parts = raw_name.strip().split()
    return " ".join(parts[:2])

def parse_sent_timestamp_from_match(groups):
    """Parse timestamp from message"""
    month, day, year, timepart, ampm = groups
    time_str = f"{timepart} {ampm}"
    source = f"{month} {day} {year} {time_str}"
    for fmt in ("%B %d %Y %I:%M:%S %p", "%B %d %Y %I:%M %p"):
        try:
            dt = datetime.strptime(source, fmt)
            date_str = dt.strftime("%Y-%m-%d")
            return date_str, dt
        except:
            continue
    return None, None

def extract_events_from_message(msg):
    """Extract login/logout/break events from a message"""
    body = get_body_text(msg)
    if not body:
        return []

    events = []
    raw_from = decode_header_value(msg.get("From") or "")
    sender = parseaddr(raw_from)[1]

    # Define all phrases to search for
    phrases_to_search = [
        (LOGIN_PHRASE, "login"),
        (LOGOUT_PHRASE, "logout"),
        (LUNCH_START_PHRASE, "lunch_start"),
        (LUNCH_END_PHRASE, "lunch_end")
    ]

    for phrase, typ in phrases_to_search:
        for m in re.finditer(re.escape(phrase), body, flags=re.I):
            start_idx = m.start()
            end_idx = m.end()

            left_window = body[max(0, start_idx-500): start_idx]
            sent_matches = list(SENT_RE.finditer(left_window))
            sent_match = sent_matches[-1] if sent_matches else None

            if not sent_match:
                prior_body = body[:start_idx]
                sent_matches2 = list(SENT_RE.finditer(prior_body))
                sent_match = sent_matches2[-1] if sent_matches2 else None

            if not sent_match:
                continue

            date_str, dt_obj = parse_sent_timestamp_from_match(sent_match.groups())
            if date_str is None:
                continue

            name_match = NAME_RE.search(body[end_idx: end_idx+300])
            if not name_match:
                name_match = NAME_RE.search(body[max(0, start_idx-200): start_idx+200])
            if not name_match:
                name_match = NAME_RE.search(body)
            name = clean_name(name_match.group(1)) if name_match else ""

            time_formatted = dt_obj.strftime("%H:%M")  # Formato 24h estandarizado

            events.append({
                "sender": sender,
                "name": normalize_name(name),
                "date": date_str,
                "dt": dt_obj,
                "time": time_formatted,
                "type": typ
            })
    return events

def calculate_break_duration_hhmm(start_time, end_time):
    """Calculate break duration in HH:MM format"""
    if not start_time or not end_time:
        return None
    
    try:
        start_parts = start_time.split(':')
        end_parts = end_time.split(':')
        
        start_minutes = int(start_parts[0]) * 60 + int(start_parts[1])
        end_minutes = int(end_parts[0]) * 60 + int(end_parts[1])
        
        duration_minutes = end_minutes - start_minutes
        
        # Handle case where lunch crosses midnight (unlikely but just in case)
        if duration_minutes < 0:
            duration_minutes += 24 * 60
        
        # Convert to HH:MM format
        hours = duration_minutes // 60
        minutes = duration_minutes % 60
        
        return f"{hours:02d}:{minutes:02d}"
    except:
        return None

def compare_break_duration(break_time_str):
    """
    Return class & label for break duration:
      - <= 60 min -> break-normal (white)
      - 61..65 min -> break-warning (yellow)
      - >= 66 min -> break-exceeded (red)
    """
    if not break_time_str or break_time_str.strip() == '':
        return 'break-normal', ''
    try:
        hh, mm = map(int, break_time_str.split(':'))
        total = hh * 60 + mm
        if total <= 60:
            return 'break-normal', 'Within limit'
        elif total <= 65:
            return 'break-warning', 'Slightly over (≤5 min)'
        else:
            return 'break-exceeded', 'Exceeds 5 minutes'
    except:
        return 'break-normal', ''

def process_mbox_file(mbox_path):
    """Process Google Chat archive (.mbox) into a dataframe with login/logout and lunch duration."""
    try:
        mbox = mailbox.mbox(mbox_path)
        grouped = {}

        for msg in mbox:
            try:
                events = extract_events_from_message(msg)
            except Exception:
                continue

            for ev in events:
                key = (ev["sender"], ev["name"], ev["date"])
                if key not in grouped:
                    grouped[key] = {
                        "login_dt": None, "login_time": "",
                        "logout_dt": None, "logout_time": "",
                        "lunch_start_dt": None, "lunch_start_time": "",
                        "lunch_end_dt": None, "lunch_end_time": ""
                    }

                if ev["type"] == "login":
                    cur = grouped[key]["login_dt"]
                    if cur is None or ev["dt"] < cur:
                        grouped[key]["login_dt"] = ev["dt"]
                        grouped[key]["login_time"] = ev["time"]

                elif ev["type"] == "logout":
                    cur = grouped[key]["logout_dt"]
                    if cur is None or ev["dt"] > cur:
                        grouped[key]["logout_dt"] = ev["dt"]
                        grouped[key]["logout_time"] = ev["time"]

                elif ev["type"] == "lunch_start":
                    cur = grouped[key]["lunch_start_dt"]
                    if cur is None or ev["dt"] < cur:
                        grouped[key]["lunch_start_dt"] = ev["dt"]
                        grouped[key]["lunch_start_time"] = ev["time"]

                elif ev["type"] == "lunch_end":
                    cur = grouped[key]["lunch_end_dt"]
                    if cur is None or ev["dt"] > cur:
                        grouped[key]["lunch_end_dt"] = ev["dt"]
                        grouped[key]["lunch_end_time"] = ev["time"]

        # Build dataframe rows
        data = []
        for (sender, name, date), times in sorted(grouped.items()):
            break_duration_hhmm = calculate_break_duration_hhmm(
                times["lunch_start_time"], times["lunch_end_time"]
            )
            data.append({
                "sender_email": sender,
                "name": normalize_name(name),
                "date": date,
                "login_time": times["login_time"],
                "logout_time": times["logout_time"],
                "lunch_start_time": times["lunch_start_time"],
                "lunch_end_time": times["lunch_end_time"],
                "break_duration": break_duration_hhmm if break_duration_hhmm else ""
            })

        return pd.DataFrame(data)

    except Exception as e:
        print(f"❌ Error processing mbox: {e}")
        return pd.DataFrame()

# Employee schedules database - Add more employees as needed
EMPLOYEE_SCHEDULES = {
    'DAVID PAREDES': '08:30-17:30',  # Formato 24h estandarizado
    # Add more employees here:
    # 'JOHN DOE': '09:00-18:00',
    # 'JANE SMITH': '07:30-16:30',
}

def standardize_time_format(time_str):
    """Convert any time format to HH:MM (24h format)"""
    if not time_str or time_str.strip() == '':
        return ''
    
    time_str = str(time_str).strip().upper()
    
    # Si ya está en formato HH:MM, verificar y devolver
    if re.match(r'^\d{1,2}:\d{2}$', time_str):
        try:
            parts = time_str.split(':')
            hour = int(parts[0])
            minute = int(parts[1])
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                return f"{hour:02d}:{minute:02d}"
        except:
            pass
    
    try:
        # Intentar parsear diferentes formatos
        formats_to_try = [
            '%I:%M %p',      # 8:30 AM
            '%I:%M%p',       # 8:30AM
            '%H:%M',         # 08:30 (24h)
            '%I.%M %p',      # 8.30 AM
            '%I.%M%p',       # 8.30AM
            '%I%p',          # 8AM
            '%I %p',         # 8 AM
        ]
        
        for fmt in formats_to_try:
            try:
                dt = datetime.strptime(time_str, fmt)
                return dt.strftime('%H:%M')
            except:
                continue
                
        # Si nada funciona, intentar extraer números
        numbers = re.findall(r'\d+', time_str)
        if len(numbers) >= 1:
            hour = int(numbers[0])
            minute = int(numbers[1]) if len(numbers) > 1 else 0
            
            # Ajustar AM/PM si está presente
            if 'PM' in time_str and hour != 12:
                hour += 12
            elif 'AM' in time_str and hour == 12:
                hour = 0
                
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                return f"{hour:02d}:{minute:02d}"
        
        return time_str  # Devolver original si no se puede convertir
    except:
        return time_str

def get_employee_schedule_times(employee_name):
    """Get the start and end times for an employee (separated)"""
    normalized_name = normalize_name(employee_name)
    schedule_str = EMPLOYEE_SCHEDULES.get(normalized_name, '08:00-17:00')
    
    if '-' in schedule_str:
        start_time, end_time = schedule_str.split('-', 1)
        return standardize_time_format(start_time.strip()), standardize_time_format(end_time.strip())
    else:
        return '08:00', '17:00'  # Default

def parse_time_to_minutes(time_str):
    """Convert time string in HH:MM format to minutes since midnight"""
    if not time_str or time_str.strip() == '':
        return None
    
    try:
        time_str = standardize_time_format(time_str)
        if ':' in time_str:
            parts = time_str.split(':')
            hour = int(parts[0])
            minute = int(parts[1])
            return hour * 60 + minute
    except:
        pass
    return None

def compare_times(actual_time, scheduled_time_str):
    """Compare actual time with scheduled time and return color class (for ENTRY times)"""
    if not actual_time or not scheduled_time_str:
        return 'no-time', ''
    
    actual_standardized = standardize_time_format(actual_time)
    scheduled_standardized = standardize_time_format(scheduled_time_str)
    
    actual_minutes = parse_time_to_minutes(actual_standardized)
    scheduled_minutes = parse_time_to_minutes(scheduled_standardized)
    
    if actual_minutes is None or scheduled_minutes is None:
        return 'no-time', ''
    
    difference = actual_minutes - scheduled_minutes
    
    if difference == 0:
        return 'on-time', 'Exact time'
    elif -5 <= difference <= 5:
        if difference < 0:
            return 'early', f'{abs(difference)} min early'
        else:
            return 'late-minor', f'{difference} min late'
    else:
        if difference < 0:
            return 'very-early', f'{abs(difference)} min early'
        else:
            return 'late-major', f'{difference} min late'

def compare_exit_times(actual_time, scheduled_time_str):
    """Compare actual exit time with scheduled exit time (DIFFERENT LOGIC FOR EXIT)"""
    if not actual_time or not scheduled_time_str:
        return 'no-time', ''
    
    actual_standardized = standardize_time_format(actual_time)
    scheduled_standardized = standardize_time_format(scheduled_time_str)
    
    actual_minutes = parse_time_to_minutes(actual_standardized)
    scheduled_minutes = parse_time_to_minutes(scheduled_standardized)
    
    if actual_minutes is None or scheduled_minutes is None:
        return 'no-time', ''
    
    difference = actual_minutes - scheduled_minutes
    
    if difference == 0:
        # Tiempo exacto - VERDE
        return 'on-time', 'Exact time'
    elif difference > 0:
        # Salió después del horario - AZUL (no importa)
        return 'early', f'{difference} min after'
    elif -5 <= difference < 0:
        # Salió 1-5 min antes - AMARILLO (aceptable)
        return 'late-minor', f'{abs(difference)} min early'
    else:
        # Salió 6+ min antes - ROJO (problemático)
        return 'late-major', f'{abs(difference)} min early'

def process_files_and_generate_table(csv_path, mbox_path):
    """Process both files and generate unified table"""
    # Process CSV
    df_csv = process_csv_file(csv_path)
    if df_csv is None:
        return None, "Error processing CSV"

    # Process MBOX
    df_mbox = process_mbox_file(mbox_path)
    if df_mbox is None or len(df_mbox) == 0:
        return None, "Error processing MBOX"

    # Create unified table
    unified_data = []
    matches_found = 0

    print(f"\n🔍 PERFORMING FLEXIBLE MATCHING (name + date):")
    print(f"   • CSV records: {len(df_csv)}")
    print(f"   • MBOX records: {len(df_mbox)}")

    # For each CSV record, find match in MBOX
    for _, csv_row in df_csv.iterrows():
        csv_name = str(csv_row['Employee']).strip()
        csv_date = csv_row['daydate']
        csv_in = csv_row['In']
        csv_out = csv_row['out']
        csv_unpaid_break = csv_row['unpaid_break']

        # Get employee schedule (separated)
        schedule_start, schedule_end = get_employee_schedule_times(csv_name)
        
        # Standardize CSV times
        csv_in_standardized = standardize_time_format(csv_in)
        csv_out_standardized = standardize_time_format(csv_out)

        # Compare entry times and get color classes (NORMAL LOGIC)
        csv_in_class, csv_in_status = compare_times(csv_in_standardized, schedule_start)
        
        # Compare exit times using EXIT LOGIC
        csv_out_class, csv_out_status = compare_exit_times(csv_out_standardized, schedule_end)

        # Check break duration color for TCW break
        tcw_break_class, tcw_break_status = compare_break_duration(csv_unpaid_break)

        # Search for match in MBOX
        mbox_match = None
        for _, mbox_row in df_mbox.iterrows():
            mbox_name = str(mbox_row['name']).strip()
            mbox_date = mbox_row['date']

            if exact_match(csv_name, csv_date, mbox_name, mbox_date):
                mbox_match = mbox_row
                matches_found += 1
                break

        # Add to unified table
        if mbox_match is not None:
            mbox_login_standardized = standardize_time_format(mbox_match['login_time'])
            mbox_logout_standardized = standardize_time_format(mbox_match['logout_time'])
            
            # Entry times use NORMAL logic
            mbox_login_class, mbox_login_status = compare_times(mbox_login_standardized, schedule_start)
            # Exit times use EXIT logic
            mbox_logout_class, mbox_logout_status = compare_exit_times(mbox_logout_standardized, schedule_end)
            
            # Check MBOX break duration color
            mbox_break_class, mbox_break_status = compare_break_duration(mbox_match['break_duration'])
            
            unified_data.append({
                'name': csv_name,
                'date': mbox_match['date'],
                'schedule_start': schedule_start,
                'schedule_end': schedule_end,
                'csv_in': csv_in_standardized,
                'csv_in_class': csv_in_class,
                'csv_in_status': csv_in_status,
                'mbox_login': mbox_login_standardized,
                'mbox_login_class': mbox_login_class,
                'mbox_login_status': mbox_login_status,
                'csv_out': csv_out_standardized,
                'csv_out_class': csv_out_class,
                'csv_out_status': csv_out_status,
                'mbox_logout': mbox_logout_standardized,
                'mbox_logout_class': mbox_logout_class,
                'mbox_logout_status': mbox_logout_status,
                'tcw_break': csv_unpaid_break,
                'tcw_break_class': tcw_break_class,
                'tcw_break_status': tcw_break_status,
                'mbox_break': mbox_match['break_duration'],
                'mbox_break_class': mbox_break_class,
                'mbox_break_status': mbox_break_status,
                'status': 'Match',
                'status_class': 'success'
            })
        else:
            unified_data.append({
                'name': csv_name,
                'date': 'Not found',
                'schedule_start': schedule_start,
                'schedule_end': schedule_end,
                'csv_in': csv_in_standardized,
                'csv_in_class': csv_in_class,
                'csv_in_status': csv_in_status,
                'mbox_login': '',
                'mbox_login_class': 'no-time',
                'mbox_login_status': '',
                'csv_out': csv_out_standardized,
                'csv_out_class': csv_out_class,
                'csv_out_status': csv_out_status,
                'mbox_logout': '',
                'mbox_logout_class': 'no-time',
                'mbox_logout_status': '',
                'tcw_break': csv_unpaid_break,
                'tcw_break_class': tcw_break_class,
                'tcw_break_status': tcw_break_status,
                'mbox_break': '',
                'mbox_break_class': 'break-normal',
                'mbox_break_status': '',
                'status': 'No match',
                'status_class': 'danger'
            })

    # Create statistics
    stats = {
        'total_csv': len(df_csv),
        'total_mbox': len(df_mbox),
        'matches_found': matches_found,
        'match_percentage': round((matches_found/len(df_csv)*100), 1) if len(df_csv) > 0 else 0
    }

    return unified_data, stats

# ==================== FLASK ROUTES ====================

@app.route('/')
def index():
    """Main page with upload form"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Process uploaded files"""
    try:
        # Check that both files were uploaded
        if 'csv_file' not in request.files or 'mbox_file' not in request.files:
            flash('Please upload both files: CSV and MBOX', 'error')
            return redirect(url_for('index'))

        csv_file = request.files['csv_file']
        mbox_file = request.files['mbox_file']

        if csv_file.filename == '' or mbox_file.filename == '':
            flash('Please select both files', 'error')
            return redirect(url_for('index'))

        if not (allowed_file(csv_file.filename) and allowed_file(mbox_file.filename)):
            flash('Invalid file types. Use CSV and MBOX', 'error')
            return redirect(url_for('index'))

        # Save files
        csv_filename = secure_filename(csv_file.filename)
        mbox_filename = secure_filename(mbox_file.filename)
        
        csv_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_filename)
        mbox_path = os.path.join(app.config['UPLOAD_FOLDER'], mbox_filename)
        
        csv_file.save(csv_path)
        mbox_file.save(mbox_path)

        # Process files
        results, stats = process_files_and_generate_table(csv_path, mbox_path)
        
        if results is None:
            flash(f'Error processing files: {stats}', 'error')
            return redirect(url_for('index'))

        # Show results
        return render_template('results.html', 
                             results=results, 
                             stats=stats,
                             csv_filename=csv_filename,
                             mbox_filename=mbox_filename)

    except Exception as e:
        print(f"Error in upload: {e}")
        flash(f'Error processing files: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/ping')
def api_ping():
    """Test endpoint"""
    return jsonify({'status': 'ok', 'message': 'Server is running'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)