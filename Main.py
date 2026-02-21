from tkinter import *
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import hashlib
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

from PIL import Image, ImageTk, ImageDraw, ImageFont

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier

from tinydb import TinyDB, Query

main = Tk()
main.geometry("1300x900")
main.title("Vision-Based Terrain Classification for Autonomous Robots")


global X, Y, X_train, X_test, y_train, y_test
global base_model, le, categories, dataset_path

MODEL_FOLDER = "model"
FEATURE_FILE = os.path.join(MODEL_FOLDER, "MobileNetV2_Features.npz")
os.makedirs(MODEL_FOLDER, exist_ok=True)


screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()

bg_img = Image.open("background.jpg")
bg_img = bg_img.resize((screen_width, screen_height), Image.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_img)
bg_label = Label(main, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)
bg_label.lower()


title = Label(
    main,
    text="Vision-Based Terrain Classification for Autonomous Outdoor Robot Navigation",
    font=('times', 20, 'bold'),
    bg='lightblue',
    fg='black'
)
title.place(x=150, y=10)

text = Text(main, height=25, width=85, font=('times', 12, 'bold'))
text.place(x=350, y=180)

def clear_text():
    text.delete('1.0', END)


base_model = MobileNetV2(weights='imagenet', include_top=False)



def uploadDataset():
    clear_text()
    global dataset_path, categories

    dataset_path = filedialog.askdirectory()
    categories = [d for d in os.listdir(dataset_path)
                  if os.path.isdir(os.path.join(dataset_path, d))]

    text.insert(END, "Dataset Loaded Successfully\n")
    text.insert(END, f"Detected Classes: {categories}\n")


def MobileNetV2_feature_extraction():
    clear_text()
    global X, Y, categories, dataset_path, base_model

    FEATURE_FILE = os.path.join("model", "MobileNetV2_Features.npz")

    if os.path.exists(FEATURE_FILE):
        data = np.load(FEATURE_FILE, allow_pickle=True)
        X = data["features"]
        Y = data["labels"]

        text.insert(END, "Loaded saved MobileNetV2 features\n")
        text.insert(END, f"Total Samples : {X.shape[0]}\n")
        text.insert(END, f"Feature Size  : {X.shape[1]}\n")
        return

    X = []
    Y = []

    text.insert(END, "MobileNetV2 Feature Extraction Started...\n")

    for cls in categories:
        cls_path = os.path.join(dataset_path, cls)

        for img_file in os.listdir(cls_path):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):

                img_path = os.path.join(cls_path, img_file)

                img = image.load_img(img_path, target_size=(64, 64))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)

                img_array = preprocess_input(img_array)

                features = base_model.predict(img_array, verbose=0)
                features = np.mean(features, axis=(1, 2))  


                X.append(features)
                Y.append(cls)

    X = np.array(X)
    Y = np.array(Y)

    np.savez_compressed(
        FEATURE_FILE,
        features=X,
        labels=Y
    )

    text.insert(END, "MobileNetV2 Feature Extraction Completed\n")
    text.insert(END, f"Total Samples : {X.shape[0]}\n")
    text.insert(END, f"Feature Size  : {X.shape[1]}\n")




def Train_test_spliting():
    clear_text()
    global X, Y, X_train, X_test, y_train, y_test, le

    le_path = os.path.join(MODEL_FOLDER, "label_encoder.pkl")

    if os.path.exists(le_path):
        le = joblib.load(le_path)
        y_encoded = le.transform(Y)
    else:
        le = LabelEncoder()
        y_encoded = le.fit_transform(Y)
        joblib.dump(le, le_path)


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,        
        random_state=42,      
        stratify=y_encoded    
    )

    text.insert(END, f"Total Samples : {X.shape[0]}\n")
    text.insert(END, f"Train Samples : {X_train.shape[0]}\n")
    text.insert(END, f"Test  Samples : {X_test.shape[0]}\n")




def performance_evaluation(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0) * 100
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0) * 100

    text.insert(END, f"\n{name} Performance\n")
    text.insert(END, f"Accuracy : {acc:.2f}\n")
    text.insert(END, f"Precision: {prec:.2f}\n")
    text.insert(END, f"Recall   : {rec:.2f}\n")
    text.insert(END, f"F1-score : {f1:.2f}\n\n")


    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_
    )
    plt.title(name + " Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    text.insert(
        END,
        classification_report(
            y_test,
            y_pred,
            target_names=le.classes_,
            zero_division=0
        )
    )

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        n_classes = len(le.classes_)
        y_test_bin = to_categorical(y_test, num_classes=n_classes)

        plt.figure(figsize=(8, 6))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            auc_score = roc_auc_score(y_test_bin[:, i], y_prob[:, i])
            plt.plot(fpr, tpr, label=f"{le.classes_[i]} (AUC = {auc_score:.2f})")

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(name + " ROC Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()




def Model_LRC():
    clear_text()
    global X_train, X_test, y_train, y_test

    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    X_test_2d  = X_test.reshape(X_test.shape[0], -1)

    model_path = os.path.join(MODEL_FOLDER, "LRC.pkl")

    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        model = LogisticRegression(max_iter=100)
        model.fit(X_train_2d, y_train)
        joblib.dump(model, model_path)

    performance_evaluation(
        "Logistic Regression",
        model,
        X_test_2d,
        y_test
    )


def Model_NBC():
    clear_text()
    global X_train, X_test, y_train, y_test

    X_train_2d = np.abs(X_train.reshape(X_train.shape[0], -1))
    X_test_2d  = np.abs(X_test.reshape(X_test.shape[0], -1))

    model_path = os.path.join(MODEL_FOLDER, "NBC.pkl")

    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        model = MultinomialNB()
        model.fit(X_train_2d, y_train)
        joblib.dump(model, model_path)

    performance_evaluation(
        "Naive Bayes",
        model,
        X_test_2d,
        y_test
    )

def Model_Ridge():
    clear_text()
    global X_train, X_test, y_train, y_test

    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    X_test_2d  = X_test.reshape(X_test.shape[0], -1)


    model_path = os.path.join(MODEL_FOLDER, "Ridge1.pkl")

    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        model = RidgeClassifier(
            alpha=1000.0,        
            random_state=42
        )
        model.fit(X_train_2d, y_train)
        joblib.dump(model, model_path)

    performance_evaluation(
        "Ridge Classifier",
        model,
        X_test_2d,
        y_test
    )



def Model_XGBoost():
    clear_text()
    global X_train, X_test, y_train, y_test, le

    Xtr = X_train.reshape(X_train.shape[0], -1)
    Xte = X_test.reshape(X_test.shape[0], -1)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(le.classes_),
        n_estimators=250,
        max_depth=6,
        learning_rate=0.08,
        eval_metric="mlogloss",
        random_state=42
    )

    model.fit(Xtr, y_train)

    joblib.dump(model, os.path.join(MODEL_FOLDER, "XGBoost.pkl"))

    performance_evaluation(
        "XGBoost",
        model,
        Xte,
        y_test
    )


def predict():
    clear_text()
    file = filedialog.askopenfilename(
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
    )

    img = image.load_img(file, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = base_model.predict(x, verbose=0)
    features = np.mean(features, axis=(1, 2)) 
    features = features.reshape(1, -1)          

    model = joblib.load(os.path.join(MODEL_FOLDER, "XGBoost.pkl"))
    le = joblib.load(os.path.join(MODEL_FOLDER, "label_encoder.pkl"))

    pred = model.predict(features)[0]
    label = le.inverse_transform([pred])[0]

    text.insert(END, f"Predicted Class : {label}\n")

    pil_img = Image.open(file)
    draw = ImageDraw.Draw(pil_img)
    draw.text((20, 20), f"Prediction: {label}", fill=(255, 0, 0))
    plt.imshow(pil_img)
    plt.axis("off")
    plt.show()



db = TinyDB("users_db.json")
users_table = db.table("users")

def signup(role):
    def register():
        u, p = user.get(), pwd.get()
        if not u or not p:
            messagebox.showerror("Error", "Fill all fields")
            return

        hashed = hashlib.sha256(p.encode()).hexdigest()
        User = Query()

        if users_table.search((User.username == u) & (User.role == role)):
            messagebox.showerror("Error", "User already exists")
            return

        users_table.insert({"username": u, "password": hashed, "role": role})
        messagebox.showinfo("Success", "Signup successful")
        win.destroy()

    win = Toplevel(main)
    win.geometry("300x200")
    Label(win, text="Username").pack()
    user = Entry(win); user.pack()
    Label(win, text="Password").pack()
    pwd = Entry(win, show="*"); pwd.pack()
    Button(win, text="Signup", command=register).pack(pady=10)

def login(role):
    def verify():
        u, p = user.get(), pwd.get()
        hashed = hashlib.sha256(p.encode()).hexdigest()
        User = Query()

        if users_table.search((User.username == u) & (User.password == hashed) & (User.role == role)):
            messagebox.showinfo("Success", "Login successful")
            win.destroy()
            clear_buttons()
            show_admin_buttons() if role == "Admin" else show_user_buttons()
        else:
            messagebox.showerror("Error", "Invalid credentials")

    win = Toplevel(main)
    win.geometry("300x200")
    Label(win, text="Username").pack()
    user = Entry(win); user.pack()
    Label(win, text="Password").pack()
    pwd = Entry(win, show="*"); pwd.pack()
    Button(win, text="Login", command=verify).pack(pady=10)

def clear_buttons():
    for w in main.winfo_children():
        if isinstance(w, Button):
            w.destroy()
    bg_label.lower()
    title.lift()
    text.lift()

def show_admin_buttons():
    clear_buttons()
    font = ('times', 13, 'bold')
    Button(main, text="Upload Dataset", command=uploadDataset, font=font).place(x=20, y=150)
    Button(main, text="FE with MobileNetV2", command=MobileNetV2_feature_extraction, font=font).place(x=20, y=200)
    Button(main, text="Train Test Split", command=Train_test_spliting, font=font).place(x=20, y=250)
    Button(main, text="Train LRC", command=Model_LRC, font=font).place(x=20, y=300)
    Button(main, text="Train NBC", command=Model_NBC, font=font).place(x=20, y=350)
    Button(main, text="Train XGBoost", command=Model_XGBoost, font=font).place(x=20, y=450)
    Button(main, text="Train Ridge", command=Model_Ridge, font=font).place(x=20, y=400)

    Button(main, text="Logout", command=show_login_screen, bg="red", font=font).place(x=20, y=500)

def show_user_buttons():
    clear_buttons()
    font = ('times', 13, 'bold')
    Button(main, text="Predict Image", command=predict, font=font).place(x=20, y=300)
    Button(main, text="Logout", command=show_login_screen, bg="red", font=font).place(x=20, y=350)

def show_login_screen():
    clear_buttons()
    font = ('times', 14, 'bold')
    Button(main, text="Admin Signup", command=lambda: signup("Admin"), font=font).place(x=200, y=100)
    Button(main, text="User Signup", command=lambda: signup("User"), font=font).place(x=450, y=100)
    Button(main, text="Admin Login", command=lambda: login("Admin"), font=font).place(x=700, y=100)
    Button(main, text="User Login", command=lambda: login("User"), font=font).place(x=950, y=100)

show_login_screen()
main.mainloop()
