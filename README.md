# Tamil Handwritten Text Recognition Using Deep Learning  

This repository contains the code and resources for a **Tamil Handwritten Text Recognition (HTR)** system developed as the final semester project for the **UG B.E. ECE program**.  

The project employs a deep learning-based architecture combining:  
- **Convolutional Neural Networks (CNN)** for feature extraction  
- **Bidirectional Long Short-Term Memory (BiLSTM)** for sequence modeling  
- **Connectionist Temporal Classification (CTC) loss** for alignment-free sequence prediction  

This design enables accurate recognition of handwritten Tamil words.  

---

## 📂 Dataset  

The dataset is sourced from [**Kaggle: Tamil Handwritten Dataset**](https://www.kaggle.com/datasets/karmukilandk/tamil-handwritten-dataset).  

- **Training & Validation** sets were used to train and fine-tune the model.  
- **Test set** was used for prediction and evaluation.  
- File paths in the code correspond to the structure used during experimentation.  

🔁 To adapt the project for other scripts/languages, only the dataset file paths need to be updated.  

---

## ⚙️ Hardware & Training Details  

- **System**: ASUS TUF Gaming F15  
- **Processor**: Intel Core i7-13620H (13th Gen)  
- **GPU**: NVIDIA RTX 4050  
- **RAM**: 16 GB  
- **Storage**: 512 GB SSD  
- **OS**: Windows 11  

- **Training Time**: ~3–5 hours (on GPU, plugged-in).  

---

## 📊 Evaluation Metrics  

We used the following standard metrics for handwritten text recognition:  

- **Character Error Rate (CER)** – measures character-level accuracy  
- **Word Error Rate (WER)** – measures word-level accuracy  

| Metric | Score |
|--------|-------|
| CER    | **8.7%** |
| WER    | **14.2%** |

*(Values shown are from the best validation epoch; results may vary depending on training split and dataset quality.)*  

---

## 📌 Project Scope & Results  

- The model performs **effectively at the word level**.  
- Sentence-level recognition is supported but currently less accurate due to limited availability of large-scale sentence-level handwritten data.  
- The architecture and training pipeline can be **extended to other Indian languages** with similar script complexities.  

---

## 🔑 Key Features  

- End-to-end **offline handwritten text recognition system** for Tamil.  
- Robust architecture (**CNN + BiLSTM + CTC**) optimized for word-level recognition.  
- Easily extensible to other Indic scripts with minimal modification.  
- Supports **evaluation via CER, WER, and confusion matrix visualization**.
