# SMN: Painting Artistic Movement Deep Learning Classifier

SMN is a deep learning project designed to classify paintings according to their artistic movements. I developed it during Le Wagon batch 1372 in March 2024. This project evolved through various experimental phases allowing me to work with data analysis, deep learning, cloud computing, and deployment strategies.

## Project course

### 1. Initial Approaches (Feature Engineering without AI)
- **Extracting dominant colors**: Attempted to classify paintings based on a single dominant color per movement. This approach failed.
- **Expanding to five dominant colors**: Increased the number of dominant colors analyzed, but classification accuracy remained poor.
- **Key Takeaway**: Artistic movements are influenced by more than just color; texture, composition, and brush strokes play a major role.

### 2. Transition to Deep Learning (CNNs & Model Optimization)
- **Built convolutional neural networks (CNNs)**: Experimented with various architectures, dropout, regularization, max-pooling, and image augmentation technique.
- **Used Google Cloud Compute for training**: Leveraged two different virtual machines to handle parallel training and experiment with two networks at once.
- **Achieved sub-50% accuracy**: Despite improvements, the results were not satisfactory.
- **Key Takeaway**: Custom CNNs alone, without a robust dataset or pre-trained features, struggle to generalize well.

### 3. Transfer Learning with VGG16 (Breaking the 50% Barrier)
- **Integrated a pre-trained VGG16 model**: Used TensorFlow’s VGG16 and added custom dense layers.
- **Unfroze the last few layers**: Fine-tuned the model on the dataset while leveraging VGG16’s learned features.
- **Crossed the 50% accuracy mark**: Performance significantly improved.
- **Key Takeaway**: Transfer learning is a powerful tool.

### 4. Dataset Refinement (Achieving Above 70% Accuracy)
- **Manually cleaned the dataset**: Removed incorrect images and restructured the dataset.
- **Refined artistic movement categories**: Chose more distinct artistic movements to reduce overlap.
- **Trained VGG16 again**: With improved data quality, the model exceeded 70% accuracy.
- **Key Takeaway**: A well-curated dataset is just as important as model architecture.

### 5. Deployment & Web Application
- **Developed an API with FastAPI**: Designed a backend API to serve the model.
- **Containerized the model with Docker**: Ensured easy deployment across environments.
- **Deployed on Google Artifact Registry**: Hosted the model for remote inference.
- **Built a Streamlit web app**: Created a user-friendly interface where users could upload or take a picture of a painting for classification.

## Future Directions
- **Expand dataset**: Gather more paintings to improve model generalization.
- **Better data cleaning**: Automate the process with computer vision techniques.
- **Increase computational resources**: Invest in better GPUs for faster training.
- **Enhance user experience**: Use generative AI to suggest music related to artistic movements or provide textual summaries.

## Skills Gained
- **Machine Learning & Deep Learning**: CNNs, Transfer Learning (VGG16), Model Optimization
- **Cloud Computing**: Google Cloud Compute, Virtual Machine Training
- **Data Processing & Feature Engineering**: Image processing, Image classification, Data cleaning
- **API Development & Deployment**: FastAPI, Docker, Streamlit, Google Artifact Registry
- **Project Management**: Experimentation, Debugging, Refining Approaches

## Conclusion
This project was a deep dive into image classification and the power of transfer learning. It solidified my understanding of model training, cloud computing, and end-to-end deep learning deployment.


Canva presentation link:
https://www.canva.com/design/DAF_MgMabq8/YygYRcl_csJgqMNs9tV8lg/view?utm_content=DAF_MgMabq8&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h1ef9cce1b3
