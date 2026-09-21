import streamlit as st
import cv2
import numpy as np
import exifread
import piexif
from PIL import Image, ImageChops, ImageEnhance
import io
import folium
from streamlit_folium import st_folium
import google.generativeai as genai
from geopy.geocoders import Nominatim
from ultralytics import YOLO

# ReportLab Libraries for PDF
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# ==========================================
# 1. PAGE CONFIG & CUSTOM CSS (With FOOHOTO Logo)
# ==========================================
st.set_page_config(page_title="FOOHOTO - Photo Forensics Suite", layout="wide")

st.markdown("""
    <!-- Google Font Montserrat -->
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@800;900&display=swap" rel="stylesheet">
    <style>
    .stApp {
        background-color: #ffffff !important;
        color: #000000 !important;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: #000000 !important;
    }
    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        background-color: #f8f9fa !important;
        color: #000000 !important;
        border: 1px solid #000000 !important;
        border-radius: 4px;
    }
    .stButton>button, .stDownloadButton>button {
        background-color: #000000 !important;
        color: #ffffff !important;
        border: 1px solid #000000 !important;
        border-radius: 4px !important;
        font-weight: 600;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background-color: #333333 !important;
        color: #ffffff !important;
    }
    
    /* Logo Header Styling */
    .logo-container {
        display: flex;
        align-items: center;
        user-select: none;
        margin-bottom: 20px;
        font-family: 'Montserrat', sans-serif;
    }
    .logo-text {
        font-size: 48px;
        font-weight: 900;
        color: #000000;
        letter-spacing: -1px;
        text-transform: uppercase;
        line-height: 1;
    }
    .magnifier-eye {
        width: 48px;
        height: 48px;
        margin: 0 1px 0 1px;
        vertical-align: middle;
        overflow: visible;
    }
    </style>
""", unsafe_allow_html=True)

# Render SVG Logo Header
st.markdown("""
    <div class="logo-container">
        <span class="logo-text">FO</span>
        <svg class="magnifier-eye" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="45" cy="45" r="32" stroke="black" stroke-width="12" />
            <path d="M68 68 L92 92" stroke="black" stroke-width="14" stroke-linecap="round" />
            <path d="M 23 45 Q 45 28 67 45 Q 45 62 23 45 Z" fill="none" stroke="black" stroke-width="3" />
            <circle cx="45" cy="45" r="8" fill="black" />
            <circle cx="43" cy="43" r="2.5" fill="white" />
        </svg>
        <span class="logo-text">HOTO</span>
    </div>
""", unsafe_allow_html=True)


# ==========================================
# 2. CORE FUNCTIONS
# ==========================================

@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

def run_ela(image_bytes, quality=90, scale=15):
    original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    buffer = io.BytesIO()
    original.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)
    
    ela_img = ImageChops.difference(original, resaved)
    extrema = ela_img.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    scale_factor = 255.0 / max_diff
    
    return ImageEnhance.Brightness(ela_img).enhance(scale_factor * (scale / 10.0))

def analyze_steganography_lsb(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img)
    return Image.fromarray(((img_np & 1) * 255).astype(np.uint8))

def get_reverse_geocode(lat, lon):
    try:
        geolocator = Nominatim(user_agent="photo_analysis_tool")
        location = geolocator.reverse((lat, lon), timeout=3)
        return location.address if location else "Address not found."
    except Exception:
        return "Geocoding unavailable."

def strip_metadata(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    clean_img = Image.new(img.mode, img.size)
    clean_img.putdata(list(img.getdata()))
    output_io = io.BytesIO()
    clean_img.save(output_io, format="JPEG")
    return output_io.getvalue()

def get_opencv_analysis(image_bytes):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return {}

    height, width, channels = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avg_brightness = float(np.mean(gray))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return {
        "Width": width,
        "Height": height,
        "Channels": channels,
        "Aspect Ratio": f"{round(width/height, 2)}:1",
        "Brightness Score": round(avg_brightness, 2),
        "Blur Score": round(blur_score, 2),
        "Focus Status": "Sharp / Focused" if blur_score > 100 else "Soft Focus / Blurry"
    }

def convert_to_degrees(value):
    try:
        d = float(value.values[0].num) / float(value.values[0].den)
        m = float(value.values[1].num) / float(value.values[1].den)
        s = float(value.values[2].num) / float(value.values[2].den)
        return d + (m / 60.0) + (s / 3600.0)
    except Exception:
        return None

def get_exif_data(image_file):
    image_file.seek(0)
    tags = exifread.process_file(image_file, details=False)
    exif_dict, gps_data = {}, {}

    for tag, value in tags.items():
        if tag not in ('JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote'):
            exif_dict[tag] = str(value)

    if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
        lat = convert_to_degrees(tags['GPS GPSLatitude'])
        lat_ref = str(tags.get('GPS GPSLatitudeRef', 'N'))
        if lat_ref == 'S' and lat: lat = -lat

        lon = convert_to_degrees(tags['GPS GPSLongitude'])
        lon_ref = str(tags.get('GPS GPSLongitudeRef', 'E'))
        if lon_ref == 'W' and lon: lon = -lon

        if lat and lon:
            gps_data['Latitude'] = lat
            gps_data['Longitude'] = lon

    return exif_dict, gps_data

def analyze_with_gemini(cv_data, exif_data, gps_data, detected_objects, custom_api_key=None):
    api_keys = [
        "AIzaSyAsrHvNkWxH6AdoMASeJ3cf1VjegFkW-Cg",
        "AIzaSyC7Gv30W62Ob7xYywar3LrOOqO_LSKJuTI",
        "AIzaSyAxUXEPzxoqX3Plf1nSk2hfto1lutqso_I",
        "AQ.Ab8RN6KfQo28FRiEK1P2vd11UwOQdsQPQehqNM5iGe6YMtOWaQ"
    ]
    if custom_api_key:
        api_keys.insert(0, custom_api_key)

    prompt = f"""
    Digital Forensics & OSINT Briefing:
    - OpenCV Metrics: {cv_data}
    - Objects Identified: {detected_objects}
    - Location Info: {gps_data if gps_data else 'None'}
    - Metadata Sample: {dict(list(exif_data.items())[:10])}
    
    Generate 4 brief bullet points covering Forensics Overview, Camera/Image Details, Scene Findings, and Location/OSINT.
    """

    last_error = None
    for key in api_keys:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            return model.generate_content(prompt).text
        except Exception as e:
            last_error = e
            continue

    raise Exception(f"All API Keys failed: {str(last_error)}")

def generate_pdf_report(filename, file_size, cv_data, exif_data, gps_data, physical_address, detected_objects, ai_report):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=16, leading=20, textColor=colors.black)
    heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading2'], fontSize=12, leading=15, textColor=colors.black, spaceBefore=10, spaceAfter=4)
    normal_style = ParagraphStyle('NormalStyle', parent=styles['Normal'], fontSize=9, leading=12, textColor=colors.black)
    
    elements = [
        Paragraph("<b>FOOHOTO - PHOTO FORENSICS & ANALYSIS REPORT</b>", title_style),
        Spacer(1, 10),
        Paragraph("<b>1. File Overview</b>", heading_style)
    ]
    
    file_info_data = [
        ["Attribute", "Value"],
        ["File Name", str(filename)],
        ["File Size", f"{round(file_size/1024, 2)} KB"],
        ["Resolution", f"{cv_data.get('Width', 'N/A')} x {cv_data.get('Height', 'N/A')}"],
        ["Blur / Focus", str(cv_data.get('Focus Status', 'N/A'))]
    ]
    t1 = Table(file_info_data, colWidths=[150, 350])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('FONTSIZE', 9),
    ]))
    elements.extend([t1, Spacer(1, 10)])
    
    # Location
    elements.append(Paragraph("<b>2. Location Data</b>", heading_style))
    lat, lon = gps_data.get('Latitude', 'N/A'), gps_data.get('Longitude', 'N/A')
    elements.extend([Paragraph(f"<b>Coordinates:</b> Lat {lat}, Long {lon}<br/><b>Address:</b> {physical_address}", normal_style), Spacer(1, 10)])

    # EXIF
    elements.append(Paragraph("<b>3. Key Metadata</b>", heading_style))
    if exif_data:
        exif_table_data = [["EXIF Tag", "Value"]] + [[str(k)[:30], str(v)[:60]] for k, v in list(exif_data.items())[:12]]
        t2 = Table(exif_table_data, colWidths=[200, 300])
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('FONTSIZE', 8),
        ]))
        elements.append(t2)
    else:
        elements.append(Paragraph("No EXIF tags found.", normal_style))
    elements.append(Spacer(1, 10))

    # AI Report
    if ai_report:
        elements.append(Paragraph("<b>4. AI Intelligence Summary</b>", heading_style))
        elements.append(Paragraph(ai_report.replace('\n', '<br/>'), normal_style))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


# ==========================================
# 3. STREAMLIT APP (UI)
# ==========================================

st.sidebar.header("Settings")
gemini_api_key = st.sidebar.text_input("Custom Gemini API Key (Optional)", type="password")

uploaded_file = st.file_uploader("Upload Image (JPG/JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image_bytes, caption="Uploaded Image", use_container_width=True)
    with col2:
        st.subheader("Quick Metadata")
        st.write(f"**Filename:** {uploaded_file.name}")
        st.write(f"**Size:** {round(len(image_bytes)/1024, 2)} KB")
        
        cv_data = get_opencv_analysis(image_bytes)
        exif_data, gps_data = get_exif_data(io.BytesIO(image_bytes))
        
        st.json(cv_data)
        if st.button("Strip Metadata (Privacy Mode)"):
            clean_bytes = strip_metadata(image_bytes)
            st.download_button("Download Clean Image", clean_bytes, "clean.jpg", "image/jpeg")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Forensics & Tampering",
        "GPS Location & Maps",
        "Metadata & EXIF Editor",
        "AI Report & PDF Download"
    ])

    # TAB 1: Forensics
    with tab1:
        st.subheader("Image Integrity Analysis")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Error Level Analysis (ELA)**")
            st.image(run_ela(image_bytes), use_container_width=True)
        with c2:
            st.write("**Least Significant Bit (LSB)**")
            st.image(analyze_steganography_lsb(image_bytes), use_container_width=True)

        st.markdown("---")
        if st.checkbox("Run Object Detection (YOLOv8)"):
            with st.spinner("Detecting objects..."):
                try:
                    yolo_model = load_yolo()
                    img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    results = yolo_model(img_pil)
                    st.image(results[0].plot(), caption="Detected Entities", use_container_width=True)
                except Exception as e:
                    st.error(f"YOLO Error: {e}")

    # TAB 2: Location
    with tab2:
        st.subheader("Geolocation Reconnaissance")
        if gps_data and "Latitude" in gps_data:
            lat, lon = gps_data["Latitude"], gps_data["Longitude"]
            physical_address = get_reverse_geocode(lat, lon)
            st.success(f"Coordinates: Lat {lat}, Long {lon}")
            st.write(f"**Address:** {physical_address}")
            
            m = folium.Map(location=[lat, lon], zoom_start=15)
            folium.Marker([lat, lon], popup=physical_address).add_to(m)
            st_folium(m, width=700, height=300)
        else:
            st.warning("No embedded GPS tags found in this image.")

    # TAB 3: Metadata Editor
    with tab3:
        st.subheader("EXIF Metadata Management")
        if exif_data:
            with st.expander("View Raw EXIF Tags", expanded=False):
                st.json(exif_data)
        
        st.write("**Modify EXIF Headers**")
        with st.form("edit_form"):
            ca, cb = st.columns(2)
            with ca:
                make = st.text_input("Make", "Camera")
                model = st.text_input("Model", "Model X")
            with cb:
                artist = st.text_input("Artist", "Investigator")
                cpy = st.text_input("Copyright", "Public Domain")
            
            if st.form_submit_button("Save & Update EXIF"):
                im = Image.open(io.BytesIO(image_bytes))
                exif_dict = {"0th": {
                    piexif.ImageIFD.Make: make.encode('utf-8'),
                    piexif.ImageIFD.Model: model.encode('utf-8'),
                    piexif.ImageIFD.Artist: artist.encode('utf-8'),
                    piexif.ImageIFD.Copyright: cpy.encode('utf-8')
                }}
                exif_bytes = piexif.dump(exif_dict)
                output_io = io.BytesIO()
                im.convert("RGB").save(output_io, format="JPEG", exif=exif_bytes)
                st.download_button("Download Updated JPEG", output_io.getvalue(), "edited_image.jpg", "image/jpeg")

    # TAB 4: AI Report & PDF
    with tab4:
        st.subheader("Report Generation")
        
        ai_report_text = ""
        if st.button("Generate AI Forensic Briefing"):
            with st.spinner("Generating summary via Gemini..."):
                try:
                    ai_report_text = analyze_with_gemini(
                        cv_data, exif_data, gps_data, "Not Scanned", gemini_api_key if gemini_api_key else None
                    )
                    st.markdown(ai_report_text)
                except Exception as e:
                    st.error(f"Gemini API Error: {str(e)}")

        st.markdown("---")
        if st.button("Download Complete PDF Report"):
            with st.spinner("Compiling PDF..."):
                physical_addr = get_reverse_geocode(gps_data["Latitude"], gps_data["Longitude"]) if (gps_data and "Latitude" in gps_data) else "N/A"
                pdf_bytes = generate_pdf_report(
                    filename=uploaded_file.name,
                    file_size=len(image_bytes),
                    cv_data=cv_data,
                    exif_data=exif_data,
                    gps_data=gps_data,
                    physical_address=physical_addr,
                    detected_objects="None/Standard",
                    ai_report=ai_report_text
                )
                st.download_button("Save Forensic Report (.PDF)", pdf_bytes, f"report_{uploaded_file.name}.pdf", "application/pdf")
