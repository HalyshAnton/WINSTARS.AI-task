# MNIST Classification 

This project implements three different architectures for classifying MNIST handwritten digits:  
1. **Random Forest (RF)** – A classical machine learning approach using decision trees  
2. **Feed Forward Neural Network (NN)** – A fully connected deep learning model  
3. **Convolutional Neural Network (CNN)** – A deep learning model optimized for image data  

The models can be trained on the MNIST dataset, evaluated, and used for individual image predictions.

---

## 🛠 Installation  

Install required dependencies using:  
```bash
pip install -r requirements.txt
```

Ensure you have `Python 3.8+` installed.

---

## 📌 Training a Model  

You can train any of the three models using the following command:  

```bash
python train.py --algorithm <model_type> --mnist-dir <path_to_dataset> --epochs 10 --batch 32 --lr 0.001 --save-path <model_file>
```

If `--mnist-dir` is not provided, the dataset will be downloaded automatically.

---

## 🖼 Making Predictions  

To classify a single image, run:  
```bash
python test.py --algorithm <model_type> --model-path <model_file> --img <image_path> --save-path <result_json>
```
Predicted probabilities will be written to json file

```json
{
    "digit.png": [0.01, 0.02, 0.03, 0.85, 0.04, 0.02, 0.01, 0.01, 0.01, 0.00]
}
```
The highest probability corresponds to digit **3**.

---

## 📊 Model Evaluation  

Each model was trained and evaluated on the MNIST test set. Below is a detailed analysis of their performance, including their strengths and weaknesses in recognizing specific digits.  

### **Random Forest (RF)**   
- **Accuracy:** **96.5%**  
- **Weaknesses:**  
  - Struggles with digits that have straight lines, such as **1, 2, and 7**.  
  - Also misclassifies numbers with rounded segments, like **3 and 8**.   

![](results/confussion%20matrix%20rf.png))

![](results/predictions%20rf.png)

### **Feed Forward Neural Network (NN)**   
- **Accuracy:** **93.7%**  
- **Weaknesses:**  
  - Frequently misclassifies **4 and 9**, often confusing them with each other.
  - Struggles with complex patterns due to limited spatial awareness.  

![](results/confussion%20matrix%20nn.png)

![](results/predictions%20nn.png)

### **Convolutional Neural Network (CNN)**   
- **Accuracy:** **98.1%** (Best performance)  
- **Weaknesses:**  
  - Sometimes confuses **7 with 4 and 9** due to their similar angles.  
  - Also misclassifies **9 and 8**, likely due to their rounded shapes.  
  - Despite these weaknesses, CNN achieves the highest accuracy and best overall performance.  

![](results/confussion%20matrix%20cnn.png)

![](results/predictions%20cnn.png)

### **Summary**  
| Model | Strengths                                    | Weaknesses |  
|--------|----------------------------------------------|----------------------------------------------------|  
| **RF** | Fast training, decent accuracy               | Misclassifies 1, 2, 7, 3, and 8 |  
| **NN** | Lowest accuracy | Struggles with 4 and 9 |  
| **CNN** | Best accuracy, captures spatial features     | Misclassifies 7 with 4 and 9, and sometimes 9 with 8 |  

CNN is the most accurate but computationally expensive. If speed is critical, RF is a good alternative, but it struggles with certain numbers. NN is not recommended.  

---

## 🔍 Notes  
- Ensure that the input image is **28x28 grayscale** for best results.  
- The dataset will be automatically downloaded if not provided.  
- Random Forest does not require a GPU, while NN and CNN benefit from one.  

---

## 📜 License  
This project is open-source and can be modified as needed.  
