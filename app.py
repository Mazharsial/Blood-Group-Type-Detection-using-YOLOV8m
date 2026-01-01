import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, send_file
from fpdf import FPDF
import qrcode
import onnxruntime as ort
from datetime import datetime
 
app = Flask(__name__)
app.secret_key = "super_secret_key"

# ---------- CONFIGURATION ----------
MODEL_PATH = r"C:\Users\Mazhar Sial\OneDrive\Desktop\dip_project\best.onnx"
LOGO_PATH = r"D:\SEMESTER 4\PYTHON Filnal project\logo.png"

model_session = ort.InferenceSession(MODEL_PATH)
class_names = ["a", "b", "d"]

UPLOAD_FOLDER = "static/uploads/"
OUTPUT_FOLDER = "static/outputs/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ------------ HELPER: INFERENCE ------------
def prepare_input(image):
    img = cv2.resize(image, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

def process_output(raw_output, img_w, img_h):
    predictions = np.squeeze(raw_output[0]).transpose()

    boxes = []
    scores = []
    class_ids = []

    # Auto-detect normalized coordinates
    is_normalized = np.max(predictions[:, :4]) <= 1.0
    model_dim = 640.0

    x_scale = img_w / model_dim
    y_scale = img_h / model_dim

    for row in predictions:
        classes_scores = row[4:]
        _, max_score, _, max_class_loc = cv2.minMaxLoc(classes_scores)
        class_id = max_class_loc[1]

        if max_score >= 0.25: 
            xc, yc, w, h = row[0], row[1], row[2], row[3]

            if is_normalized:
                xc *= model_dim
                yc *= model_dim
                w *= model_dim
                h *= model_dim

            x1 = int((xc - w/2) * x_scale)
            y1 = int((yc - h/2) * y_scale)
            width = int(w * x_scale)
            height = int(h * y_scale)
            
            boxes.append([x1, y1, width, height])
            scores.append(float(max_score))
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.25, nms_threshold=0.5)

    final_detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            x = max(0, x)
            y = max(0, y)
            
            final_detections.append({
                "box": [x, y, x+w, y+h], 
                "score": scores[i],
                "class_id": class_ids[i]
            })

    return final_detections

# ------------ HELPER: PDF GENERATION (COMPACT FORMAT) ------------
def generate_pdf_report(data):
    pdf_filename = f"report_{int(datetime.now().timestamp())}.pdf"
    pdf_path = os.path.join(OUTPUT_FOLDER, pdf_filename)
    
    # 1. Generate QR Code
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr_data = f"Name: {data['name']}, Blood: {data['blood_group']}, Date: {datetime.now()}"
    qr.add_data(qr_data)
    qr.make(fit=True)
    qr_img = qr.make_image(fill='black', back_color='white')
    qr_path = os.path.join(OUTPUT_FOLDER, "temp_qr.png")
    qr_img.save(qr_path)

    # 2. Create PDF
    pdf = FPDF()
    pdf.add_page()
    
    # --- HEADER ---
    if os.path.exists(LOGO_PATH):
        try: pdf.image(LOGO_PATH, x=10, y=8, w=25) 
        except: pass

    # Title & Contact Info (Reduced line height from 10 to 6)
    pdf.set_y(10) # Start text at top
    pdf.set_font("Arial", 'B', size=16)
    pdf.cell(0, 8, "Sial Medical Center", ln=True, align="C")
    
    pdf.set_font("Arial", 'I', size=10) # Smaller font for address
    pdf.cell(0, 6, "Address: Chowk Manga Mandi, Lahore, Pakistan", ln=True, align="C")
    pdf.cell(0, 6, "Contact: +92 3044335771 | Email: sial@medicalcenter.com", ln=True, align="C")
    
    pdf.ln(5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y()) # Add a horizontal line separator
    pdf.ln(5)
    
    # --- PATIENT INFO ---
    pdf.set_font("Arial", 'B', size=12)
    # Using cell height 7 instead of 10
    pdf.cell(0, 7, f"Patient Name: {data['name']}", ln=True)
    pdf.cell(0, 7, f"Patient Age: {data['age']} years", ln=True)
    pdf.cell(0, 7, f"Patient Gender: {data['gender']}", ln=True)
    pdf.ln(3)

    # --- RESULTS ---
    pdf.set_text_color(200, 0, 0) # Red color for result
    pdf.cell(0, 7, f"Blood Group Detected: {data['blood_group']}", ln=True)
    pdf.set_text_color(0, 0, 0) # Reset to black
    
    pdf.cell(0, 7, f"Detected Components: {data['labels'].upper()}", ln=True)
    pdf.ln(3)

    # --- METHODOLOGY ---
    pdf.cell(0, 7, "Test Methodology:", ln=True)
    pdf.set_font("Arial", size=10) 
    pdf.multi_cell(0, 5, "The blood group detection was performed using YOLOv8 object detection model. The system analyzes the agglutination patterns to determine the blood type.")
    pdf.ln(2)
    pdf.multi_cell(0, 5, "Conclusion: The detected blood group is provided above based on antigen reaction.")
    pdf.ln(5) 

    # --- IMAGES ---
    y_images = pdf.get_y()
    
    # Check if page break needed
    if y_images > 220:
        pdf.add_page()
        y_images = 20

    # QR Code 
    pdf.image(qr_path, x=160, y=y_images, w=35)
    
    # Blood Image (Left) - Height fixed to 50 to prevent taking too much space
    if os.path.exists(data['image_path']):
         pdf.image(data['image_path'], x=10, y=y_images, h=50)

    
    pdf.set_y(y_images + 55)

    # --- SIGNATURE ---
    pdf.cell(0, 10, "Doctor's Signature: ___________________", ln=True, align="L")
    
    pdf.set_font("Arial", 'I', size=8)
    pdf.cell(0, 10, "Generated by Sial Tech AI System", ln=True, align="C")

    pdf.output(pdf_path)
    return pdf_filename


# ------------ ROUTES ----------------

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username == "admin" and password == "1234":
            session["logged_in"] = True
            return redirect(url_for("patient_info"))
        return render_template("login.html", error="Invalid Credentials")
    return render_template("login.html")

@app.route("/patient", methods=["GET", "POST"])
def patient_info():
    if not session.get("logged_in"): return redirect(url_for("login"))
    if request.method == "POST":
        session["name"] = request.form["name"]
        session["age"] = request.form["age"]
        session["gender"] = request.form["gender"]
        session["cnic"] = request.form.get("cnic", "N/A")
        return redirect(url_for("upload_file"))
    return render_template("patient_info.html")

@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if not session.get("logged_in"): return redirect(url_for("login"))

    if request.method == "POST":
        file = request.files["file"]
        if file.filename == '': return redirect(request.url)

        timestamp = int(datetime.now().timestamp())
        filename = f"input_{timestamp}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # 1. Inference
        img = cv2.imread(filepath)
        h, w = img.shape[:2]
        blob = prepare_input(img)
        input_name = model_session.get_inputs()[0].name
        raw = model_session.run(None, {input_name: blob})
        detections = process_output(raw, w, h)

        detected_labels = []
        
        # 2. Draw Boxes
        for d in detections:
            idx = int(d["class_id"])
            if 0 <= idx < len(class_names):
                label_name = class_names[idx]
                detected_labels.append(label_name)
                
                color = (0, 255, 0)
                if label_name == 'a': color = (255, 0, 0)
                elif label_name == 'b': color = (0, 255, 255)
                elif label_name == 'd': color = (0, 0, 255)

                x1, y1, x2, y2 = map(int, d["box"])
                score_txt = f"{label_name.upper()} {d['score']:.2f}"

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                (text_w, text_h), _ = cv2.getTextSize(score_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (x1, y1 - 25), (x1 + text_w, y1), color, -1)
                cv2.putText(img, score_txt, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        unique_labels = set(detected_labels)
        
        # 3. Blood Group Logic
        bg = "Unknown"
        if unique_labels == {"a", "d"}: bg = "A+ (A Positive)"
        elif unique_labels == {"a"}: bg = "A- (A Negative)"
        elif unique_labels == {"b", "d"}: bg = "B+ (B Positive)"
        elif unique_labels == {"b"}: bg = "B- (B Negative)"
        elif unique_labels == {"a", "b", "d"}: bg = "AB+ (AB Positive)"
        elif unique_labels == {"a", "b"}: bg = "AB- (AB Negative)"
        elif unique_labels == {"d"}: bg = "O+ (O Positive)"
        elif len(unique_labels) == 0: bg = "O- (O Negative)"

        out_filename = f"detected_{timestamp}.jpg"
        out_path = os.path.join(OUTPUT_FOLDER, out_filename)
        cv2.imwrite(out_path, img) 

        session["blood_group"] = bg
        session["labels"] = ", ".join(unique_labels)
        session["image_path"] = out_path

        # 4. PDF Generation
        data = {
            "name": session.get("name"),
            "age": session.get("age"),
            "gender": session.get("gender"),
            "blood_group": bg,
            "labels": ", ".join(unique_labels),
            "image_path": out_path
        }
        pdf_name = generate_pdf_report(data)
        session["pdf_filename"] = pdf_name

        return redirect(url_for("result"))

    return render_template("upload.html")

@app.route("/result")
def result():
    if not session.get("logged_in"): return redirect(url_for("login"))
    return render_template("result.html", 
                           name=session.get("name"),
                           age=session.get("age"),
                           gender=session.get("gender"),
                           blood_group=session.get("blood_group"),
                           labels=session.get("labels"),
                           image=session.get("image_path"))

@app.route("/download")
def download_pdf():
    pdf_file = session.get("pdf_filename")
    if pdf_file:
        return send_file(os.path.join(OUTPUT_FOLDER, pdf_file), as_attachment=True)
    return "Report not found", 404

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)