import React, { useState } from "react";
import "./App.css";
import { Button } from "react-bootstrap";

// ################# component for navigation bar ##########
const NavigationBar = ({ activeTab, setActiveTab }) => {
  return (
    <nav className="navigation-bar">
      <Button
        variant={activeTab === "inference" ? "primary" : "light"}
        onClick={() => setActiveTab("inference")}
      >
        Inference
      </Button>
      <Button
        variant={activeTab === "statistics" ? "primary" : "light"}
        onClick={() => setActiveTab("statistics")}
      >
        Statistics
      </Button>
      <Button
        variant={activeTab === "api" ? "primary" : "light"}
        onClick={() => setActiveTab("api")}
      >
        API
      </Button>
      <Button
        variant={activeTab === "mlflow" ? "primary" : "light"}
        onClick={() => setActiveTab("mlflow")}
      >
        MLflow
      </Button>
    </nav>
  );
};

// #################################################################### inference components ############
// ################# component for image selection #########
const FileUpload = ({ onFileSelect }) => {
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      onFileSelect(file);
    }
  };

  return (
    <div className="file-upload">
      <input
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        id="file-input"
        style={{ display: "none" }}
      />
      <label htmlFor="file-input">
        <Button variant="primary" as="span">
          Upload Image
        </Button>
      </label>
    </div>
  );
};

// ####################### component for label-dropdown ####
const LabelSelect = ({ selectedLabel, onLabelChange }) => {
  return (
    <div className="label-select">
      <select
        value={selectedLabel}
        onChange={(e) => onLabelChange(Number(e.target.value))}
      >
        <option value={0}>NORMAL</option>
        <option value={1}>PNEUMONIA</option>
      </select>
    </div>
  );
};

// ############### component for image preview #########
const ImagePreview = ({ file }) => {
  if (!file) {
    return <div className="image-preview">No image selected</div>;
  }

  return (
    <div className="image-preview">
      <img
        src={URL.createObjectURL(file)}
        alt="Preview"
        style={{ maxWidth: "100%", maxHeight: "300px" }}
      />
    </div>
  );
};

// ################################ predict-button ###########
const PredictButton = ({ onPredict, isDisabled }) => {
  return (
    <button
      onClick={onPredict}
      disabled={isDisabled}
      className="predict-button"
    >
      Predict
    </button>
  );
};

// ############################ prediction item ###############
const PredictionItem = ({ label, value }) => {
  const formatPrediction = (value) => {
    return (
      " pneumonia indication " + (parseFloat(value) * 100).toFixed(2) + "%"
    );
  };

  return (
    <div className="prediction-item">
      <span className="prediction-label">{label}:</span>
      <span className="prediction-value">{formatPrediction(value)}</span>
    </div>
  );
};

// ############################ prediction results ############
const PredictionResults = ({ prediction, isLoading }) => {
  return (
    <div className="prediction-results-container">
      <h2 className="prediction-results-title">Prediction Results</h2>
      <div className="prediction-results-content">
        {isLoading ? (
          <div className="loading-spinner"></div>
        ) : prediction ? (
          <>
            <PredictionItem
              label="Champion"
              value={prediction["prediction champion"]}
            />
            <PredictionItem
              label="Challenger"
              value={prediction["prediction challenger"]}
            />
            <PredictionItem
              label="Baseline"
              value={prediction["prediction baseline"]}
            />
          </>
        ) : (
          <p>No prediction results yet. Click 'Predict' to start.</p>
        )}
      </div>
    </div>
  );
};

// ############### component for inference container #########
const InferenceViewer = ({
  selectedImage,
  setSelectedImage,
  label,
  setLabel,
  prediction,
  isLoading,
  handlePrediction,
}) => {
  return (
    <div className="inference-container">
      <h1 className="header-title">Xray Pneumonia Detection</h1>
      <div className="inference-content-container">
        <div className="left-column">
          <div className="input-container">
            <FileUpload onFileSelect={setSelectedImage} />
            <LabelSelect selectedLabel={label} onLabelChange={setLabel} />
            <PredictButton
              onPredict={handlePrediction}
              isDisabled={!selectedImage || label === null}
            />
          </div>
          <ImagePreview file={selectedImage} />
        </div>
        <div className="right-column">
          <PredictionResults prediction={prediction} isLoading={isLoading} />
        </div>
      </div>
    </div>
  );
};

// #################################################################### statistics integration ###############
// #################### Input component for model comparison plot #######
const ModelComparisonInput = ({ value, onChange }) => {
  return (
    <input
      type="number"
      value={value}
      onChange={onChange}
      placeholder="Enter a positive integer"
      min="1"
      className="statistics-input" // Füge eine CSS-Klasse hinzu
    />
  );
};

// #################### Button to show model comparison plot ############
const ShowModelComparisonButton = ({ onClick }) => {
  return (
    <Button onClick={onClick} className="statistics-button">
      Show Model Comparison
    </Button>
  );
};

// #################### Component for model comparison plot ############
const ModelComparisonPlot = ({ plotUrl }) => {
  return (
    <div className="plot-container">
      {plotUrl ? (
        <img
          src={plotUrl}
          alt="Model Comparison Plot"
          style={{ maxWidth: "100%", maxHeight: "500px" }}
        />
      ) : (
        <p>No plot to display yet.</p>
      )}
    </div>
  );
};

// #################### Input component for confusion matrix #######
const ConfusionMatrixInput = ({ value, onChange }) => {
  return (
    <input
      type="number"
      value={value}
      onChange={onChange}
      placeholder="Enter a positive integer"
      min="1"
      className="statistics-input" // Verwende dieselbe CSS-Klasse wie beim anderen Input
    />
  );
};

// #################### Button component for confusion matrix #######
const ShowConfusionMatrixButton = ({ onClick }) => {
  return (
    <Button onClick={onClick} className="statistics-button">
      Show Confusion Matrix
    </Button>
  );
};

// #################### Component for confusion matrixplot  ############
const ConfusionMatrixPlot = ({ matrixUrl }) => {
  return (
    <div className="plot-container">
      {matrixUrl ? (
        <img
          src={matrixUrl}
          alt="Confusion Matrix Plot"
          style={{
            width: "auto",
            height: "auto",
          }}
        />
      ) : (
        <p>No plot to display yet.</p>
      )}
    </div>
  );
};

// #################### wrapper component for all statistics ############
const StatisticsViewer = ({}) => {
  const [inputValue, setInputValue] = useState("50");
  const [plotUrl, setPlotUrl] = useState(null);
  const [matrixValue, setMatrixValue] = useState("50");
  const [matrixUrl, setMatrixUrl] = useState(null);

  const handleInputChange = (event) => {
    const value = event.target.value;
    // Überprüfen, ob der Wert eine positive ganze Zahl ist
    if (/^\d+$/.test(value) || value === "") {
      setInputValue(value);
    }
  };

  const handleMatrixInputChange = (event) => {
    const value = event.target.value;
    // Überprüfen, ob der Wert eine positive ganze Zahl ist
    if (/^\d+$/.test(value) || value === "") {
      setMatrixValue(value);
    }
  };

  const fetchComparisonPlot = async () => {
    const windowSize = parseInt(inputValue, 10);

    if (isNaN(windowSize) || windowSize <= 0) {
      alert("Please enter a valid positive integer.");
      return;
    }

    try {
      const response = await fetch(
        `http://127.0.0.1:8000/get_comparsion_plot?window=${windowSize}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Da der Endpoint ein Bild zurückgibt, erstellen wir eine URL daraus
      const imageBlob = await response.blob();
      const imageURL = URL.createObjectURL(imageBlob);
      setPlotUrl(imageURL);
    } catch (error) {
      console.error("Error fetching plot:", error);
      alert("Failed to fetch plot.");
    }
  };

  const fetchConfusionMatrix = async () => {
    const matrixSize = parseInt(matrixValue, 10);

    if (isNaN(matrixSize) || matrixSize <= 0) {
      alert("Please enter a valid positive integer for the matrix.");
      return;
    }

    try {
      const response = await fetch(
        `http://127.0.0.1:8000/get_confusion_matrix_plot?window=${matrixSize}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const imageBlob = await response.blob();
      const imageURL = URL.createObjectURL(imageBlob);
      setMatrixUrl(imageURL);
    } catch (error) {
      console.error("Error fetching confusion matrix:", error);
      alert("Failed to fetch confusion matrix.");
    }
  };

  return (
    <div className="statistics-container">
      <h1 className="header-title">Xray Pneumonia Detection</h1>
      <div className="statistics-content-container">
        <div className="left-column">
          <div className="control-container-plot">
            <span className="window-param-setting">Enter window parameter</span>
            <ModelComparisonInput
              value={inputValue}
              onChange={handleInputChange}
            />
            <ShowModelComparisonButton onClick={fetchComparisonPlot} />
          </div>
          <ModelComparisonPlot plotUrl={plotUrl} />
        </div>
        <div className="right-column">
          <div className="control-container-matrix">
            <span className="window-param-setting">
              Enter last n predictions
            </span>
            <ConfusionMatrixInput
              value={matrixValue}
              onChange={handleMatrixInputChange}
            />
            <ShowConfusionMatrixButton onClick={fetchConfusionMatrix} />
          </div>
          <ConfusionMatrixPlot matrixUrl={matrixUrl} />
        </div>
      </div>
    </div>
  );
};

// #################################################################### API GUI integration component ############
const APIDocsViewer = () => {
  return (
    <div className="external-gui-container">
      <iframe
        src="http://localhost:8000/docs"
        title="FastAPI Documentation"
        style={{ width: "100%", height: "100%", border: "none" }}
      />
    </div>
  );
};

// #################################################################### MLFlow GUI integration component ############
const MLFlowViewer = () => {
  return (
    <div className="external-gui-container">
      <iframe
        src="http://localhost:8080"
        title="MLFlow GUI"
        style={{ width: "100%", height: "100%", border: "none" }}
      />
    </div>
  );
};

// ############################## APP ###########################
// ##############################################################
// ##############################################################

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [label, setLabel] = useState(0); // Default to NEGATIVE (0)
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("inference");

  const handlePrediction = async () => {
    if (!selectedImage || label === null) {
      alert("Please select an image and a label.");
      return;
    }

    setIsLoading(true);
    setPrediction(null);

    try {
      // Create FormData and append the file and label
      const formData = new FormData();
      formData.append("file", selectedImage);
      formData.append("label", label);

      console.log("Label-Typ vor dem Senden:", typeof label, label);

      // Send the request to the new endpoint
      const response = await fetch(
        "http://127.0.0.1:8000/upload_image_from_frontend",
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setPrediction(result);
    } catch (error) {
      console.error("Error:", error);
      setPrediction({ error: "Error occurred during prediction" });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <NavigationBar activeTab={activeTab} setActiveTab={setActiveTab} />
        <div className="integration-container">
          {activeTab === "inference" && (
            <InferenceViewer
              selectedImage={selectedImage}
              setSelectedImage={setSelectedImage}
              label={label}
              setLabel={setLabel}
              prediction={prediction}
              isLoading={isLoading}
              handlePrediction={handlePrediction}
            />
          )}
          {activeTab === "statistics" && <StatisticsViewer />}
          {activeTab === "api" && <APIDocsViewer />}
          {activeTab === "mlflow" && <MLFlowViewer />}
        </div>
      </header>
    </div>
  );
}

export default App;
