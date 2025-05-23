Real-time Voice Cleaner with GRUUNet2 & Streamlit

A Real-time Voice Cleaning application using a PyTorch deep learning model (GRUUNet2) and an interactive web interface built with Streamlit. This project aims to reduce noise from microphone audio input in real-time, similar to noise suppression features in video conferencing applications.


(Replace the placeholder above with a link to a screenshot or GIF demo of your application)
✨ Key Features

*    Real-time Voice Cleaning: Processes audio from the microphone directly.

*    GRUUNet2 Deep Learning Model: Utilizes a U-Net architecture with GRU cells for sequential processing and effective denoising.

*    Interactive User Interface: Built with Streamlit for ease of use, including start/stop controls.

*    Client-Side Processing (WebRTC): Uses streamlit-webrtc to capture audio from the user's browser.

*    Settings Visualization: Displays the model configuration and STFT parameters used.

🛠️ Technologies Used

*    Python 3.x

*    PyTorch: For building and running the GRUUNet2 deep learning model.

*    Streamlit: For creating the interactive web user interface.

*    streamlit-webrtc: For handling real-time audio streaming from the browser.

*    Librosa & Torchaudio: For audio processing (STFT, Mel Spectrogram, resampling).

*    NumPy: For numerical array manipulation.

*    AV (PyAV): For handling audio frames within streamlit-webrtc.

*    SoundFile: For reading/writing audio files (e.g., for downloading processed audio).

🧠 Model Architecture

The core model used is GRUUNet2, a U-Net variant modified by incorporating Gated Recurrent Unit (GRU) cells at its bottleneck. This allows the model not only to learn feature representations from the frequency domain (via the U-Net's convolutional parts) but also to capture temporal dependencies in the audio signal through the GRUs.

The input to the model is the Mel Log-Magnitude Spectrogram of the audio, and the model is trained to predict the difference between the input (noisy) spectrogram and the target (clean) spectrogram.
⚙️ Setup and Installation

Ensure you have Python 3.8+ and pip installed.

Clone the Repository:

    git clone [https://github.com/belacks/audio-denoising](https://github.com/belacks/audio-denoising) 
    cd audio-denoising 

Create and Activate a Virtual Environment (Recommended):

    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate

Install Dependencies:
Create a requirements.txt file in your project root with the following content (you can copy and paste this). These are the core dependencies for this specific audio denoising project. Then, install them using pip:

    pip install -r requirements.txt

Content for requirements.txt:

    streamlit==1.43.1
    numpy==2.1.3
    torch==2.6.0
    torchaudio==2.6.0
    librosa==0.11.0
    streamlit-webrtc==0.62.4
    av==14.3.0
    soundfile==0.13.1
    # Note: pip will automatically handle other necessary sub-dependencies.
    # For example, librosa might pull in audioread, scipy, numba (optional), etc.
    # streamlit-webrtc will pull in aiortc, pylibsrtp, etc.

Model Checkpoint Placement:

*    Ensure your model file (checkpoint.pth) is placed according to the path defined in the Streamlit script (e.g., saves/GRUUNet2-dari_tult/checkpoint.pth relative to your Streamlit application file).

*    Ensure the model architecture definition file (e.g., gruunet2.py) and utility files (e.g., utils.py if used by gruunet2.py or the app) are in the same directory as your Streamlit application file (e.g., app_streamlit_realtime.py).

▶️ How to Run the Application

Once all dependencies are installed and the model is correctly placed, run the Streamlit application from your terminal:

streamlit run app_streamlit_realtime.py

The application will open in your default web browser. Allow microphone access when prompted by the browser.
📂 Project Structure (Example)

├── app_streamlit_realtime.py  # Main Streamlit application script
├── gruunet2.py                # GRUUNet2 model architecture definition
├── utils.py                   # Utility functions (if used by gruunet2.py or app)
├── main.ipynb                 # Jupyter Notebook for model training (optional, for reference)
├── saves/
│   └── GRUUNet2-dari_tult/
│       └── checkpoint.pth     # Trained model weights file
├── requirements.txt           # Python dependencies list (cleaned version)
└── README.md                  # This file

🚀 Future Enhancements

*    Further latency optimization for a better real-time experience.

*    Exploration of more advanced phase reconstruction techniques besides Griffin-Lim.

*    Addition of options to select different denoising models.

*    Real-time visualization of input and output spectrograms.

*    Testing with various noise types and acoustic conditions.

🙏 Acknowledgements

*    Mention if this project was inspired by specific research papers, public datasets, or other repositories.
