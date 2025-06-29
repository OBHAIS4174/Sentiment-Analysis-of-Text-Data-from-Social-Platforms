# Sentiment Analysis Project

A machine learning-powered sentiment analysis application built with Flask that analyzes text sentiment and provides real-time predictions through a web interface.

## Features

- **Real-time Sentiment Analysis**: Analyze text sentiment with instant results
- **Web Interface**: User-friendly Flask web application
- **Machine Learning Model**: Custom trained model for accurate sentiment prediction
- **RESTful API**: Easy integration with other applications
- **Interactive Dashboard**: Visual representation of sentiment analysis results

## Prerequisites

- Python 3.7 or higher
- pip package manager
- Git (for cloning the repository)

## Installation & Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/OBHAIS4174/Sentiment-Analysis-of-Text-Data-from-Social-Platforms>
cd sentiment-analysis-project
```

### Step 2: Setting up the Virtual Environment

1. **Install virtualenv** (if not already installed):
```bash
pip install virtualenv
```

2. **Create a virtual environment**:
```bash
virtualenv venv
```

### Step 3: Activate the Virtual Environment

**On Windows:**
```bash
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

### Step 4: Install Dependencies

Install all required libraries:
```bash
pip install -r requirements.txt
```

### Step 5: Train the Model

1. Ensure all training files are in the project directory:
   - `intents.json` - Training data with sentiment examples
   - `train.py` - Model training script
   - `model.py` - Model architecture definition
   - `nltk_utils.py` - Natural language processing utilities

2. **Run the training script**:
```bash
python train.py
```

3. After training completes, a model file (`data.pth`) will be generated

### Step 6: Run the Application

1. **Start the Flask application**:
```bash
python app.py
```

2. **Open your web browser** and navigate to:
```
http://127.0.0.1:5000
```

## Project Structure

```
sentiment-analysis-project/
│
├── app.py                 # Flask web application
├── train.py              # Model training script
├── model.py              # Neural network model definition
├── nltk_utils.py         # NLP preprocessing utilities
├── intents.json          # Training data and sentiment examples
├── requirements.txt      # Python dependencies
├── data.pth             # Trained model file (generated after training)
├── venv/                # Virtual environment directory
├── static/              # CSS, JavaScript, images
├── templates/           # HTML templates
└── README.md           # Project documentation
```

## Usage

### Web Interface
1. Navigate to `http://127.0.0.1:5000` in your browser
2. Enter text in the input field
3. Click "Analyze Sentiment" to get results
4. View sentiment classification and confidence scores

### API Endpoints
- `POST /predict` - Analyze sentiment of provided text
- `GET /` - Main application interface

## Model Information

- **Architecture**: Neural Network with NLTK preprocessing
- **Training Data**: Custom sentiment dataset in `intents.json`
- **Features**: Tokenization, stemming, bag-of-words representation
- **Output**: Sentiment classification with confidence scores

## Customization

### Adding New Training Data
1. Edit `intents.json` to add new sentiment examples
2. Re-run the training process:
```bash
python train.py
```

### Modifying the Model
- Edit `model.py` to change neural network architecture
- Adjust hyperparameters in `train.py`
- Retrain the model after modifications

## Troubleshooting

**Common Issues:**

1. **ModuleNotFoundError**: Ensure virtual environment is activated and dependencies are installed
2. **NLTK Data Missing**: Run `python -c "import nltk; nltk.download('punkt')"`
3. **Port Already in Use**: Change port in `app.py` or kill existing processes
4. **Model File Not Found**: Ensure `train.py` has been run successfully

## Dependencies

Key libraries used in this project:
- Flask - Web framework
- PyTorch - Machine learning framework
- NLTK - Natural language processing
- NumPy - Numerical computations
- JSON - Data handling

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or support, please contact [your-email@example.com]

---

**Note**: Make sure to activate your virtual environment before running any commands. Deactivate it when done using:
```bash
deactivate
```
