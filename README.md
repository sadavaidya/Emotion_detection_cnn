🎭 Emotion Detection Using CNN

This project implements an emotion detection model using Convolutional Neural Networks (CNNs) to classify facial expressions into distinct emotions like happiness, sadness, anger, and more.
Discover how deep learning can recognize and interpret emotions from images!

Emotion detection is a key technology in fields like:

🎭 Sentiment analysis

🧑‍⚕️ Mental health monitoring

🤖 Human-computer interaction

This project leverages a CNN-based approach to analyze facial images and predict the emotion expressed. With a robust dataset and optimized architecture, this model demonstrates the power of AI in understanding human expressions.


📂 Dataset

The dataset used for this project is sourced from Kaggle and includes labeled images for the following emotions:

 Happy
 
 Sad
 
 Angry
 
 Fearful
 
 Surprised
 
 Neutral
 
👉 Download the Dataset Here (https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer)

⚙️ Installation

Follow these steps to get started:

Clone this repository:

git clone  https://github.com/sadavaidya/Emotion_detection_cnn.git 

cd emotion-detection  

Download the dataset:

Download the dataset from the link above and place it in the data/ directory.

🚀 Usage

Preprocess and train the dataset:

python Pre_preprocessing_and_training.py  

Test the model:

python test.py  

Make predictions on a custom image:

python prediction.py --image_path path_to_image.jpg  



📸 Example Prediction

Below is an example of the model predicting the emotion "Surprise" :

real_time_img_emotion_surprised.png


🤝 Contributing

💡 Ideas or Suggestions?

Feel free to fork the repository, open an issue, or submit a pull request. Contributions are always welcome!


