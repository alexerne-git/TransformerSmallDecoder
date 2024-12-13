## **Small Language Model: Decoder Architecture with Shakespeare Dataset**

The **goal** of this project is to implement a **transformer-based, character-level language model (GPT-like)** and train it on the **Shakespeare dataset**. Configurations for running the model are provided for both **Google Colab** and a **Cluster environment**.
For detailed documentation about the Transformer Decoder Architecture and a step-by-step visual illustration, please check the [Transformer Decoder documentation](https://alexerne-git.github.io/TransformerSmallDecoder/).

---

### **1. Expected Results:**

The model should generate a Shakespearean text as shown below:

**Input Sequence:**
```
O God, O God!
```
**Generated Text:**

```
O God, O God!

GLOUCESTER:
Prantagently, do that I know: consul
In the gates of appeal's a deep, that treater.

LADY CAPULET:
Romeo, come, my goods
Madam, my lord, all at greatermandancely.

Nurse:
Their wulls it subject a man; gentleman?

LADY CAPULET:
Thy lord? Saint Bolingbroke, is meet us the morrow;
For Gloucester's lord, your speak pention;
Let's so return blind him your day to-day.

HASTINGS:
Good's duke much forth, as I would say take there is.

QUEEN MARGARET:
For I may pay Bolingbroke it at with him.

```

> **Note:** When prompting the text and asking ChatGPT the style of the text we got this answer: *"This text resembles a pseudo-Shakespearean style, imitating the structure and diction of Elizabethan English but with nonsensical or fabricated language."* - close enough considering our GPT has only about 19 million parameters (compared to 175 Billion for GPT-3).


----

### **2. Training, Evaluation, Experiments:**

The best parameters used for the model were the followings:

| **Parameter**             | **Value** |
|----------------------------|-----------|
| **Batch size**             | 128       |
| **Block size**             | 128       |
| **Maximum iterations**     | 1400      |
| **Learning rate**          | 3e-4      |
| **Evaluation iterations**  | 100       |
| **Embedding size (n_embd)**| 512       |
| **Number of heads (n_head)**| 8         |
| **Number of layers (n_layer)**| 6        |
| **Dropout rate**           | 0.2       |

---


**Results at Optimal Point:**

- **Training Loss**: 1.22
- **Validation Loss**: 1.62
- **Perplexity**: 5.52
- **Training Time (min)**: 18  


**Experiments** Detailed results and experiments on different parameters can be found [here](./Documentation/experiments.md).

**Pytorch functions** A Notebook is provided that computes formulas used in the code with simplified examples, to understand how the pytorch functions work, this can be found on [this Notebook](./Documentation/pytorch_notebook.ipynb)

----

### **3. Setup Instructions:**

Both codes for Google Colab and the Cluster can be found in [this folder](./Code/readme.md)

#### **Running on Google Colab**
1. Open [this Google Colab notebook](./Code/google_colab_code.ipynb).
2. Upload the [Shakespeare dataset](./Data/dataset.txt)
3. Run the cells to install dependencies, train, and evaluate the model.

### **Running on a Cluster**
1. Follow the instructions in the [Cluster Folder](./Code/Cluster/readme.md)
---
