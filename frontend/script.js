const API_BASE = 'http://localhost:8000';

document.getElementById('disease-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const fileInput = document.getElementById('image-upload');
    const resultDiv = document.getElementById('disease-result');
    
    if (!fileInput.files[0]) {
        showResult(resultDiv, 'Please select an image', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    try {
        const response = await fetch(`${API_BASE}/predict-disease`, {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const result = await response.json();
            showResult(resultDiv, `Disease: ${result.disease} (Confidence: ${(result.confidence * 100).toFixed(2)}%)`, 'success');
        } else {
            throw new Error('Prediction failed');
        }
    } catch (error) {
        showResult(resultDiv, 'Error analyzing image', 'error');
    }
});

document.getElementById('recommendation-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const resultDiv = document.getElementById('recommendation-result');
    
    const data = {
        nitrogen: parseFloat(document.getElementById('nitrogen').value),
        phosphorus: parseFloat(document.getElementById('phosphorus').value),
        potassium: parseFloat(document.getElementById('potassium').value),
        temperature: parseFloat(document.getElementById('temperature').value),
        humidity: parseFloat(document.getElementById('humidity').value),
        ph: parseFloat(document.getElementById('ph').value),
        rainfall: parseFloat(document.getElementById('rainfall').value)
    };
    
    try {
        const response = await fetch(`${API_BASE}/recommend-crop`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        if (response.ok) {
            const result = await response.json();
            showResult(resultDiv, `Recommended Crop: ${result.recommended_crop}`, 'success');
        } else {
            throw new Error('Recommendation failed');
        }
    } catch (error) {
        showResult(resultDiv, 'Error getting recommendation', 'error');
    }
});

function showResult(div, message, type) {
    div.textContent = message;
    div.className = `result ${type}`;
    div.style.display = 'block';
}