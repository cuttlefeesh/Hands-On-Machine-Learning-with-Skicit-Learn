# Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow (2nd Edition)

**Penulis:** Aurélien Géron

Buku "Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow" edisi ke-2 ini adalah panduan komprehensif yang dirancang untuk membantu pembaca memahami dan mengimplementasikan konsep-konsep *Machine Learning* dan *Deep Learning* secara praktis. Dengan pendekatan "belajar sambil melakukan", buku ini menggabungkan teori minimal dengan banyak contoh kode Python yang siap produksi, menggunakan *framework* populer seperti Scikit-Learn, Keras, dan TensorFlow 2.

Buku ini cocok untuk programmer yang ingin beralih ke Machine Learning, mahasiswa, atau siapa saja yang ingin memperdalam pemahaman mereka tentang membangun sistem cerdas.

## Isi Buku Secara General

Buku ini dibagi menjadi dua bagian besar:

**Bagian 1: Fondasi Machine Learning (Bab 1-8)**
Bagian ini memperkenalkan konsep-konsep dasar *Machine Learning*, mulai dari jenis-jenis masalah ML, hingga algoritma tradisional seperti Regresi Linier, Support Vector Machines (SVMs), dan Decision Trees. Pembaca akan belajar tentang persiapan data, pemilihan model, evaluasi kinerja, dan teknik regularisasi.

**Bagian 2: Deep Learning (Bab 9-19)**
Bagian ini menyelami dunia *Deep Learning*, dimulai dari pengenalan Jaringan Saraf Tiruan (JST) dasar hingga arsitektur yang lebih kompleks seperti Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), Autoencoders, dan Generative Adversarial Networks (GANs). Pembahasan meliputi pelatihan JST yang dalam, pemrosesan citra, pemrosesan bahasa alami, *reinforcement learning*, serta topik praktis seperti pemuatan data, pra-pemrosesan, dan *deployment* model.

Setiap bab dilengkapi dengan latihan untuk membantu pembaca mengaplikasikan pengetahuan yang telah diperoleh.

## General Overview
### Bab 1: The Machine Learning Landscape (Lanskap Machine Learning)
Bab ini memberikan pengantar umum tentang Machine Learning, menjelaskan apa itu ML, mengapa penting, dan jenis-jenis masalah ML yang berbeda (Supervised, Unsupervised, Reinforcement Learning). Pembaca akan diperkenalkan pada konsep dasar seperti model, data, dan tujuan pembelajaran.

### Bab 2: End-to-End Machine Learning Project (Proyek Machine Learning dari Awal hingga Akhir)
Bab ini memandu pembaca melalui proyek Machine Learning yang lengkap, mulai dari mendapatkan data, eksplorasi data, persiapan data, pemilihan model, pelatihan, hingga *fine-tuning* model. Ini adalah bab yang sangat praktis yang menunjukkan alur kerja ML yang umum.

### Bab 3: Classification (Klasifikasi)
Bab ini berfokus pada salah satu tugas *supervised learning* yang paling umum: klasifikasi. Konsep-konsep seperti pengklasifikasi biner dan multikelas, metrik evaluasi kinerja (akurasi, presisi, recall, F1-score, confusion matrix, ROC curve), dan *precision/recall trade-off* dijelaskan secara mendalam.

### Bab 4: Training Models (Melatih Model)
Bab ini menyelami mekanisme internal model, khususnya model linier. Pembahasan meliputi Regresi Linier (Normal Equation dan Gradient Descent), Regresi Polinomial, serta berbagai teknik regularisasi (Ridge, Lasso, Elastic Net, Early Stopping) untuk mengatasi *overfitting*. Bab ini juga memperkenalkan Regresi Logistik dan Regresi Softmax.

### Bab 5: Support Vector Machines (SVMs)
Bab ini memperkenalkan Support Vector Machines (SVMs), sebuah model yang kuat dan serbaguna untuk klasifikasi (linier dan non-linier) serta regresi. Konsep inti seperti *large margin classification*, *soft margin*, dan *kernel trick* (termasuk Kernel Polinomial dan RBF) dijelaskan secara rinci.

### Bab 6: Decision Trees (Pohon Keputusan)
Bab ini membahas Decision Trees, model yang mudah diinterpretasikan dan serbaguna untuk klasifikasi dan regresi. Pembahasan meliputi cara kerja pohon, pelatihan, visualisasi, prediksi, serta teknik regularisasi untuk mengontrol kompleksitas pohon dan mencegah *overfitting*.

### Bab 7: Ensemble Learning and Random Forests (Pembelajaran Ensemble dan Random Forests)
Bab ini memperkenalkan *Ensemble Learning*, di mana beberapa prediktor digabungkan untuk meningkatkan kinerja. Strategi seperti Voting Classifiers, Bagging, Pasting, Random Forests, dan Boosting (termasuk AdaBoost, Gradient Boosting, dan XGBoost) dibahas secara rinci.

### Bab 8: Dimensionality Reduction (Reduksi Dimensi)
Bab ini menjelaskan pentingnya mengurangi jumlah fitur dalam dataset untuk mengatasi "kutukan dimensi". Konsep proyeksi dan *manifold learning* dibahas, bersama dengan algoritma populer seperti Principal Component Analysis (PCA), Kernel PCA, dan Locally Linear Embedding (LLE), serta t-SNE untuk visualisasi.

### Bab 9: Unsupervised Learning (Pembelajaran Tanpa Pengawasan)
Bab ini fokus pada *unsupervised learning*, di mana model belajar dari data tanpa label. Topik utama meliputi algoritma *clustering* (K-Means, DBSCAN, Gaussian Mixtures) untuk mengelompokkan data, serta deteksi anomali dan estimasi kepadatan.

### Bab 10: Introduction to Artificial Neural Networks (Pengenalan Jaringan Saraf Tiruan)
Bab ini adalah pengantar *Deep Learning*, dimulai dengan neuron biologis ke neuron buatan. Dibahas arsitektur Jaringan Saraf Tiruan (JST) dasar seperti Perceptron dan Multi-Layer Perceptrons (MLPs), algoritma Backpropagation untuk pelatihan, serta implementasi praktis menggunakan Keras API di TensorFlow 2.

### Bab 11: Training Deep Neural Networks (Melatih Jaringan Saraf Tiruan Dalam)
Bab ini membahas tantangan umum dalam melatih JST yang sangat dalam (DNN) seperti *vanishing/exploding gradients* dan *overfitting*. Berbagai solusi modern diperkenalkan, termasuk inisialisasi bobot yang lebih baik, fungsi aktivasi non-saturasi, Batch Normalization, Gradient Clipping, optimizers yang lebih cepat, dan teknik regularisasi (Dropout, L1/L2).

### Bab 12: Custom Models and Training with TensorFlow (Model Kustom dan Pelatihan dengan TensorFlow)
Bab ini menyelam lebih dalam ke TensorFlow, menjelaskan cara membangun model kustom, lapisan kustom, fungsi *loss* kustom, metrik kustom, dan bahkan *training loop* kustom. Ini memberikan kontrol yang lebih besar ketika Keras API standar tidak cukup fleksibel.

### Bab 13: Loading and Preprocessing Data with TensorFlow (Memuat dan Mempraproses Data dengan TensorFlow)
Bab ini fokus pada penanganan data skala besar dan efisien di TensorFlow. Pembahasan meliputi `tf.data` API untuk membangun *pipeline* input yang efisien, dan format TFRecord untuk penyimpanan data biner yang dioptimalkan.

### Bab 14: Deep Computer Vision Using Convolutional Neural Networks (Visi Komputer Mendalam Menggunakan Jaringan Saraf Konvolusional)
Bab ini menerapkan *Deep Learning* ke Computer Vision dengan Convolutional Neural Networks (CNNs). Struktur CNN (lapisan konvolusional, *pooling*), arsitektur populer (ResNet, Inception, Xception), dan *transfer learning* untuk gambar dibahas secara rinci.

### Bab 15: Processing Sequences Using Recurrent Neural Networks and Attention (Memproses Urutan Menggunakan Jaringan Saraf Berulang dan Atensi)
Bab ini memperkenalkan Jaringan Saraf Berulang (RNNs) untuk memproses data sekuensial seperti teks dan deret waktu. Dibahas arsitektur RNN dasar, masalah *vanishing gradients*, solusi seperti LSTM dan GRU, serta pengantar mekanisme Attention dan Transformer Networks.

### Bab 16: Processing Natural Language with RNNs and Attention (Memproses Bahasa Alami dengan RNN dan Atensi)
Bab ini adalah kelanjutan dari Bab 15, secara spesifik menerapkan RNNs dan Attention ke Natural Language Processing (NLP). Topik meliputi *word embeddings*, arsitektur *encoder-decoder* untuk penerjemahan mesin, mekanisme *Attention* yang mendalam, arsitektur Transformer, dan pengantar *Large Language Models* (LLMs).

### Bab 17: Autoencoders and GANs (Autoencoder dan GAN)
Bab ini membahas dua keluarga model *generatif*: Autoencoders dan Generative Adversarial Networks (GANs). Autoencoders untuk reduksi dimensi dan pembelajaran representasi (termasuk *denoising* dan *variational* autoencoders), sementara GANs untuk menghasilkan data baru yang realistis melalui persaingan generator dan diskriminator.

### Bab 18: Reinforcement Learning (Pembelajaran Penguatan)
Bab ini memperkenalkan Reinforcement Learning (RL), di mana agen belajar berinteraksi dengan lingkungan untuk memaksimalkan hadiah. Dibahas konsep-konsep dasar RL, kebijakan optimal, dan algoritma kunci seperti Q-Learning (DQN), Policy Gradients, Actor-Critic, dan PPO, serta penggunaan OpenAI Gym.

### Bab 19: Training and Deploying TensorFlow Models (Melatih dan Menyebarkan Model TensorFlow)
Bab ini membahas aspek praktis dari ML: melatih model dalam skala besar (strategi distribusi TensorFlow), menyebarkan model untuk inferensi (TensorFlow Serving), mengoptimalkan model untuk perangkat seluler dan *edge* (TensorFlow Lite), dan menjalankan model di browser (TensorFlow.js). Bab ini juga menyentuh MLOps dengan TFX.
